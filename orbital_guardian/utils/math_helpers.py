"""
Math helpers for spherical coordinate systems, quaternion operations, and
surface-normal utilities used throughout the Orbital Guardian game.

Key CG concepts:
  - Quaternion rotation avoids gimbal lock when orienting objects on a sphere.
  - Tangent-plane projection lets us move entities along the sphere surface.
  - Simple value-noise provides organic height variation for procedural terrain.
"""

import math
import random
from panda3d.core import Vec3, LQuaternionf, Mat4, LMatrix4f


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def normalized(v):
    """Return a normalized copy of Vec3 *v* (safe for zero-length)."""
    length = v.length()
    if length < 1e-8:
        return Vec3(0, 0, 1)
    return v / length


def lerp_vec3(a, b, t):
    """Linear interpolation between two Vec3s."""
    return a + (b - a) * t


def slerp_vec3(a, b, t):
    """Spherical linear interpolation between two unit Vec3s."""
    a = normalized(a)
    b = normalized(b)
    dot = max(-1.0, min(1.0, a.dot(b)))
    if abs(dot) > 0.9995:
        # Vectors nearly parallel — fall back to lerp
        return normalized(lerp_vec3(a, b, t))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    return Vec3(a.x * wa + b.x * wb,
                a.y * wa + b.y * wb,
                a.z * wa + b.z * wb)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_from_forward_up(forward, up):
    """
    Build a quaternion that orients an object so its local +Y axis points
    along *forward* and its local +Z axis points along *up*.

    This is the Panda3D convention: +Y = forward, +Z = up.
    """
    forward = normalized(forward)
    up = normalized(up)
    # Re-orthogonalize
    right = forward.cross(up)
    if right.length() < 1e-6:
        # forward and up are parallel — pick an arbitrary right
        right = Vec3(1, 0, 0) if abs(forward.x) < 0.9 else Vec3(0, 1, 0)
        right = normalized(forward.cross(right))
    right = normalized(right)
    up = normalized(right.cross(forward))  # NOT forward.cross(right) — Panda3D is right-handed

    # Actually we need: right cross forward should give up in a right-handed system.
    # Panda3D: X=right, Y=forward, Z=up
    # Build rotation matrix (column-major for Panda3D)
    mat = Mat4(
        right.x, right.y, right.z, 0,
        forward.x, forward.y, forward.z, 0,
        up.x, up.y, up.z, 0,
        0, 0, 0, 1
    )
    quat = LQuaternionf()
    quat.setFromMatrix(mat)
    return quat


def quat_look_at(position, target, up):
    """Return a quaternion that looks from *position* toward *target* with the
    given *up* hint."""
    forward = normalized(target - position)
    return quat_from_forward_up(forward, up)


def slerp_quat(a, b, t):
    """Spherical linear interpolation between two Panda3D quaternions."""
    # Panda3D does not expose slerp directly on LQuaternionf in all builds,
    # so we implement it here.
    dot = a.getR() * b.getR() + a.getI() * b.getI() + a.getJ() * b.getJ() + a.getK() * b.getK()
    # If dot < 0, negate one quat to take short path
    if dot < 0:
        b = LQuaternionf(-b.getR(), -b.getI(), -b.getJ(), -b.getK())
        dot = -dot
    if dot > 0.9995:
        # Very close — use linear interpolation
        result = LQuaternionf(
            a.getR() + t * (b.getR() - a.getR()),
            a.getI() + t * (b.getI() - a.getI()),
            a.getJ() + t * (b.getJ() - a.getJ()),
            a.getK() + t * (b.getK() - a.getK()),
        )
        result.normalize()
        return result
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    wa = math.sin((1 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    return LQuaternionf(
        wa * a.getR() + wb * b.getR(),
        wa * a.getI() + wb * b.getI(),
        wa * a.getJ() + wb * b.getJ(),
        wa * a.getK() + wb * b.getK(),
    )


# ---------------------------------------------------------------------------
# Spherical surface helpers
# ---------------------------------------------------------------------------

def surface_up(pos, planet_center=None):
    """Local 'up' vector for a point on a sphere centered at *planet_center*."""
    if planet_center is None:
        planet_center = Vec3(0, 0, 0)
    return normalized(pos - planet_center)


def snap_to_surface(pos, radius, height_offset=0.0, planet_center=None):
    """Project *pos* onto the sphere surface at *radius* + *height_offset*."""
    if planet_center is None:
        planet_center = Vec3(0, 0, 0)
    direction = normalized(pos - planet_center)
    return planet_center + direction * (radius + height_offset)


def tangent_frame(pos, reference_forward, planet_center=None):
    """
    Compute an orthonormal tangent frame {right, forward, up} on the sphere
    surface at *pos*.

    *reference_forward* is projected onto the tangent plane and re-orthogonalized.
    Returns (right, forward, up) as Vec3 tuple.
    """
    up = surface_up(pos, planet_center)
    # Project reference_forward onto tangent plane
    fwd = reference_forward - up * reference_forward.dot(up)
    if fwd.length() < 1e-6:
        # reference_forward is parallel to up — pick an arbitrary tangent
        fwd = Vec3(1, 0, 0) if abs(up.x) < 0.9 else Vec3(0, 1, 0)
        fwd = fwd - up * fwd.dot(up)
    fwd = normalized(fwd)
    right = up.cross(fwd)
    right = normalized(right)
    # Re-derive forward for perfect orthogonality
    fwd = normalized(right.cross(up))
    return right, fwd, up


# ---------------------------------------------------------------------------
# Simple noise for terrain generation
# ---------------------------------------------------------------------------

def _hash(ix, iy, iz):
    """Simple integer hash for 3D grid-based noise."""
    n = ix * 374761393 + iy * 668265263 + iz * 1274126177
    n = (n ^ (n >> 13)) * 1103515245
    return (n ^ (n >> 16)) & 0x7FFFFFFF


def value_noise_3d(x, y, z):
    """
    Basic 3D value noise in [0, 1].  Uses trilinear interpolation between
    hashed grid corners — no external libraries needed.

    CG concept: procedural noise is fundamental to terrain generation,
    texture synthesis, and organic-looking geometry.
    """
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    iz = int(math.floor(z))
    fx = x - ix
    fy = y - iy
    fz = z - iz

    # Smooth interpolation (Hermite / smoothstep)
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    uz = fz * fz * (3 - 2 * fz)

    def corner(dx, dy, dz):
        return (_hash(ix + dx, iy + dy, iz + dz) & 0xFFFF) / 0xFFFF

    # Trilinear interpolation of 8 corners
    c000 = corner(0, 0, 0)
    c100 = corner(1, 0, 0)
    c010 = corner(0, 1, 0)
    c110 = corner(1, 1, 0)
    c001 = corner(0, 0, 1)
    c101 = corner(1, 0, 1)
    c011 = corner(0, 1, 1)
    c111 = corner(1, 1, 1)

    x00 = c000 + ux * (c100 - c000)
    x10 = c010 + ux * (c110 - c010)
    x01 = c001 + ux * (c101 - c001)
    x11 = c011 + ux * (c111 - c011)

    y0 = x00 + uy * (x10 - x00)
    y1 = x01 + uy * (x11 - x01)

    return y0 + uz * (y1 - y0)


def fbm_3d(x, y, z, octaves=4, lacunarity=2.0, gain=0.5):
    """
    Fractal Brownian Motion — layer multiple octaves of value noise for
    richer, more natural terrain.
    """
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for _ in range(octaves):
        total += value_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude
        max_val += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / max_val


def random_point_on_sphere(radius=1.0):
    """Return a uniformly distributed random point on a sphere of given radius."""
    # Marsaglia's method
    while True:
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        s = u * u + v * v
        if s < 1.0:
            break
    factor = 2.0 * math.sqrt(1.0 - s)
    return Vec3(u * factor * radius, v * factor * radius, (1.0 - 2.0 * s) * radius)


# ---------------------------------------------------------------------------
# Easing functions for procedural animation
# ---------------------------------------------------------------------------
# CG concept: easing functions remap a linear parameter t ∈ [0,1] through
# polynomial curves to produce natural-looking acceleration and deceleration.
# These replace the jarring linearity of uniform interpolation with organic
# motion that obeys an approximation of physical inertia.

def ease_in_quad(t):
    """Quadratic ease-in: slow start, accelerating.  f(t) = t²."""
    return t * t


def ease_out_cubic(t):
    """Cubic ease-out: fast start, decelerating.  f(t) = 1 - (1-t)³."""
    u = 1.0 - t
    return 1.0 - u * u * u


def ease_in_out_quad(t):
    """
    Quadratic ease-in-out: slow start and end, fast middle.
    Piecewise quadratic that is C¹-continuous at t = 0.5.
    """
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - 2.0 * (1.0 - t) * (1.0 - t)


def ease_out_elastic(t):
    """Elastic ease-out: overshoots then settles. Good for hit reactions."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    p = 0.3
    s = p / 4.0
    return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1.0


def lerp_float(a, b, t):
    """Linear interpolation between two floats."""
    return a + (b - a) * t
