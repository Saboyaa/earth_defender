"""
Procedural mesh generation for Orbital Guardian.

All geometry is created at runtime using Panda3D's low-level Geom API —
no external model files are loaded.  This demonstrates core CG concepts:
  - Icosphere subdivision (recursive midpoint tessellation)
  - Vertex attribute packing (position, normal, color, UV)
  - Per-vertex noise displacement for organic terrain
  - Flat shading via per-face duplicate vertices
  - Primitive assembly with indexed triangle lists
"""

import math
import random
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode, NodePath,
    Vec3, Vec4, LColor,
    ShadeModelAttrib, CardMaker, TransparencyAttrib, BillboardEffect,
    GeomPoints,
)
from utils.math_helpers import normalized, fbm_3d, value_noise_3d


# -----------------------------------------------------------------------
# Color palette (hex → Vec4)
# -----------------------------------------------------------------------
def _hex(h, a=1.0):
    """Convert '#RRGGBB' to Vec4 in [0,1]."""
    h = h.lstrip('#')
    return Vec4(int(h[0:2], 16) / 255,
                int(h[2:4], 16) / 255,
                int(h[4:6], 16) / 255, a)

PLANET_GREEN  = _hex('#4a7c59')
PLANET_BROWN  = _hex('#8B6914')
PLANET_LGREEN = _hex('#6b8f71')
PLAYER_BLUE   = _hex('#3d85c6')
PLAYER_ACCENT = _hex('#e8d44d')
METEOR_ROCK   = _hex('#8b4513')
METEOR_HOT    = _hex('#ff6b35')
SPACE_DARK    = _hex('#0a0a2e')


# -----------------------------------------------------------------------
# Icosphere generation
# -----------------------------------------------------------------------

# Golden ratio
_PHI = (1.0 + math.sqrt(5.0)) / 2.0

# 12 vertices of a regular icosahedron (normalised to unit sphere)
_ICO_VERTS = [
    normalized(Vec3(-1,  _PHI, 0)),
    normalized(Vec3( 1,  _PHI, 0)),
    normalized(Vec3(-1, -_PHI, 0)),
    normalized(Vec3( 1, -_PHI, 0)),
    normalized(Vec3(0, -1,  _PHI)),
    normalized(Vec3(0,  1,  _PHI)),
    normalized(Vec3(0, -1, -_PHI)),
    normalized(Vec3(0,  1, -_PHI)),
    normalized(Vec3( _PHI, 0, -1)),
    normalized(Vec3( _PHI, 0,  1)),
    normalized(Vec3(-_PHI, 0, -1)),
    normalized(Vec3(-_PHI, 0,  1)),
]

# 20 triangular faces of the icosahedron (CCW winding)
_ICO_TRIS = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]


def _subdivide(vertices, triangles, subdivisions):
    """
    Subdivide each triangle into 4 by splitting edges at midpoints,
    projecting new vertices onto the unit sphere.

    CG concept: recursive midpoint tessellation produces an icosphere —
    a sphere approximation with near-uniform triangle sizes, far superior
    to lat/lon UV spheres which bunch triangles at the poles.
    """
    midpoint_cache = {}

    def _midpoint(i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key in midpoint_cache:
            return midpoint_cache[key]
        v1 = vertices[i1]
        v2 = vertices[i2]
        mid = normalized(Vec3(
            (v1.x + v2.x) * 0.5,
            (v1.y + v2.y) * 0.5,
            (v1.z + v2.z) * 0.5,
        ))
        idx = len(vertices)
        vertices.append(mid)
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_tris = []
        midpoint_cache.clear()
        for i0, i1, i2 in triangles:
            a = _midpoint(i0, i1)
            b = _midpoint(i1, i2)
            c = _midpoint(i2, i0)
            new_tris.append((i0, a, c))
            new_tris.append((i1, b, a))
            new_tris.append((i2, c, b))
            new_tris.append((a, b, c))
        triangles = new_tris
    return vertices, triangles


def _sphere_uv(n):
    """Compute UV from a normalised position on the unit sphere."""
    u = 0.5 + math.atan2(n.z, n.x) / (2 * math.pi)
    v = 0.5 - math.asin(max(-1, min(1, n.y))) / math.pi
    return u, v


# -----------------------------------------------------------------------
# Planet mesh
# -----------------------------------------------------------------------

def make_planet_mesh(radius=10.0, subdivisions=3, noise_scale=1.5,
                     noise_amplitude=0.4):
    """
    Generate a procedural icosphere planet with vertex-color terrain and
    slight height perturbation via fractal noise.

    Returns a NodePath containing the planet geometry.
    """
    # Build icosphere vertex list and index list
    verts = list(_ICO_VERTS)
    tris = list(_ICO_TRIS)
    verts, tris = _subdivide(verts, tris, subdivisions)

    # Compute per-vertex height offset using FBM noise
    heights = []
    for v in verts:
        n = fbm_3d(v.x * noise_scale, v.y * noise_scale, v.z * noise_scale,
                    octaves=4)
        heights.append(n)

    # --- Build Panda3D geometry with FLAT shading (duplicate verts per face) ---
    fmt = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData('planet', fmt, Geom.UHStatic)
    vdata.setNumRows(len(tris) * 3)

    writer_v = GeomVertexWriter(vdata, 'vertex')
    writer_n = GeomVertexWriter(vdata, 'normal')
    writer_c = GeomVertexWriter(vdata, 'color')
    writer_t = GeomVertexWriter(vdata, 'texcoord')

    prim = GeomTriangles(Geom.UHStatic)
    idx = 0

    palette = [PLANET_GREEN, PLANET_BROWN, PLANET_LGREEN]

    for i0, i1, i2 in tris:
        face_verts = []
        for vi in (i0, i1, i2):
            base = verts[vi]
            h = heights[vi] * noise_amplitude
            pos = base * (radius + h)
            face_verts.append(pos)

        # Flat-shading normal: cross product of two edges
        e1 = face_verts[1] - face_verts[0]
        e2 = face_verts[2] - face_verts[0]
        face_normal = normalized(e1.cross(e2))

        for k, vi in enumerate((i0, i1, i2)):
            pos = face_verts[k]
            writer_v.addData3(pos)
            writer_n.addData3(face_normal)

            # Per-vertex color based on height + noise for variety
            h_val = heights[vi]
            if h_val < 0.38:
                color = PLANET_BROWN
            elif h_val < 0.55:
                color = PLANET_GREEN
            else:
                color = PLANET_LGREEN
            # Subtle per-vertex variation
            noise_tint = value_noise_3d(
                verts[vi].x * 3.0, verts[vi].y * 3.0, verts[vi].z * 3.0
            ) * 0.1 - 0.05
            color = Vec4(
                max(0, min(1, color.x + noise_tint)),
                max(0, min(1, color.y + noise_tint)),
                max(0, min(1, color.z + noise_tint)),
                1.0
            )
            writer_c.addData4(color)

            u, v = _sphere_uv(verts[vi])
            writer_t.addData2(u, v)

        prim.addVertices(idx, idx + 1, idx + 2)
        idx += 3

    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('planet_geom')
    node.addGeom(geom)
    np = NodePath(node)
    np.setAttrib(ShadeModelAttrib.make(ShadeModelAttrib.MFlat))
    return np


# -----------------------------------------------------------------------
# Atmosphere shell (slightly larger transparent sphere)
# -----------------------------------------------------------------------

def make_atmosphere_mesh(radius=10.8, subdivisions=3):
    """
    Generate a smooth sphere for the atmosphere glow effect.
    Uses smooth (vertex) normals since the atmosphere shader relies on
    per-pixel view-direction dot products for the fresnel rim.
    """
    verts = list(_ICO_VERTS)
    tris = list(_ICO_TRIS)
    verts, tris = _subdivide(verts, tris, subdivisions)

    fmt = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData('atmosphere', fmt, Geom.UHStatic)
    vdata.setNumRows(len(verts))

    writer_v = GeomVertexWriter(vdata, 'vertex')
    writer_n = GeomVertexWriter(vdata, 'normal')
    writer_c = GeomVertexWriter(vdata, 'color')
    writer_t = GeomVertexWriter(vdata, 'texcoord')

    for v in verts:
        pos = v * radius
        writer_v.addData3(pos)
        writer_n.addData3(v)  # smooth normal = normalised position
        writer_c.addData4(0.4, 0.7, 1.0, 0.3)
        u, uv_v = _sphere_uv(v)
        writer_t.addData2(u, uv_v)

    prim = GeomTriangles(Geom.UHStatic)
    for i0, i1, i2 in tris:
        prim.addVertices(i0, i1, i2)
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('atmosphere_geom')
    node.addGeom(geom)
    return NodePath(node)


# -----------------------------------------------------------------------
# Meteor mesh (rough, jagged icosphere)
# -----------------------------------------------------------------------

def make_meteor_mesh(radius=1.0, subdivisions=1, jaggedness=0.35):
    """
    Low-poly jagged rock.  Each vertex is displaced randomly along its
    normal to create an irregular rocky surface.  Flat-shaded.
    """
    verts = list(_ICO_VERTS)
    tris = list(_ICO_TRIS)
    verts, tris = _subdivide(verts, tris, subdivisions)

    # Random radial displacement per vertex
    displaced = []
    for v in verts:
        r = radius * (1.0 + random.uniform(-jaggedness, jaggedness))
        displaced.append(v * r)

    fmt = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData('meteor', fmt, Geom.UHStatic)
    vdata.setNumRows(len(tris) * 3)

    writer_v = GeomVertexWriter(vdata, 'vertex')
    writer_n = GeomVertexWriter(vdata, 'normal')
    writer_c = GeomVertexWriter(vdata, 'color')
    writer_t = GeomVertexWriter(vdata, 'texcoord')

    prim = GeomTriangles(Geom.UHStatic)
    idx = 0

    for i0, i1, i2 in tris:
        p0, p1, p2 = displaced[i0], displaced[i1], displaced[i2]
        e1 = p1 - p0
        e2 = p2 - p0
        fn = normalized(e1.cross(e2))

        for p in (p0, p1, p2):
            writer_v.addData3(p)
            writer_n.addData3(fn)
            # Mix rock brown and hot orange randomly per face
            t = random.uniform(0.0, 1.0)
            c = Vec4(
                METEOR_ROCK.x + t * (METEOR_HOT.x - METEOR_ROCK.x),
                METEOR_ROCK.y + t * (METEOR_HOT.y - METEOR_ROCK.y),
                METEOR_ROCK.z + t * (METEOR_HOT.z - METEOR_ROCK.z),
                1.0
            )
            writer_c.addData4(c)
            writer_t.addData2(0, 0)

        prim.addVertices(idx, idx + 1, idx + 2)
        idx += 3

    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('meteor_geom')
    node.addGeom(geom)
    np = NodePath(node)
    np.setAttrib(ShadeModelAttrib.make(ShadeModelAttrib.MFlat))
    return np


# -----------------------------------------------------------------------
# Primitive geometry builders (standalone NodePath versions)
# -----------------------------------------------------------------------
# CG concept — Affine transformation hierarchy:
#   Each body part is its own NodePath in a scene-graph tree.  When a parent
#   joint rotates, the engine composes the parent's 4×4 model matrix with
#   every descendant's local matrix automatically.  This is the same
#   principle behind skeletal animation: T_world = T_root · T_hip · T_thigh · …
#   Rotations, translations, and scales (all affine transforms) propagate
#   down the hierarchy without any manual matrix multiplication in Python.

def _add_box(writer_v, writer_n, writer_c, writer_t, prim, idx,
             cx, cy, cz, sx, sy, sz, color):
    """
    Append an axis-aligned box centered at (cx,cy,cz) with half-extents
    (sx,sy,sz) to the ongoing geometry writers.  Returns new idx.

    Each face gets flat-shading normals and the supplied *color*.
    """
    # 6 faces, 2 triangles each, 3 verts per tri = 36 verts
    corners = [
        Vec3(cx - sx, cy - sy, cz - sz),
        Vec3(cx + sx, cy - sy, cz - sz),
        Vec3(cx + sx, cy + sy, cz - sz),
        Vec3(cx - sx, cy + sy, cz - sz),
        Vec3(cx - sx, cy - sy, cz + sz),
        Vec3(cx + sx, cy - sy, cz + sz),
        Vec3(cx + sx, cy + sy, cz + sz),
        Vec3(cx - sx, cy + sy, cz + sz),
    ]
    # Faces: (indices into corners, normal)
    faces = [
        ([0, 1, 2, 3], Vec3(0, 0, -1)),  # -Z
        ([5, 4, 7, 6], Vec3(0, 0,  1)),  # +Z
        ([4, 0, 3, 7], Vec3(-1, 0, 0)),  # -X
        ([1, 5, 6, 2], Vec3( 1, 0, 0)),  # +X
        ([4, 5, 1, 0], Vec3(0, -1, 0)),  # -Y
        ([3, 2, 6, 7], Vec3(0,  1, 0)),  # +Y
    ]
    for idxs, normal in faces:
        c = [corners[i] for i in idxs]
        for tri in [(0, 1, 2), (0, 2, 3)]:
            for ti in tri:
                writer_v.addData3(c[ti])
                writer_n.addData3(normal)
                writer_c.addData4(color)
                writer_t.addData2(0, 0)
            prim.addVertices(idx, idx + 1, idx + 2)
            idx += 3
    return idx


def _add_cylinder(writer_v, writer_n, writer_c, writer_t, prim, idx,
                   cx, cy, cz, radius, height, segments, color):
    """
    Append a vertical cylinder (along Z) centered at (cx, cy, cz).
    Includes top and bottom caps.  Returns new idx.
    """
    half_h = height / 2.0
    # Side faces
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        x0, y0 = math.cos(a0) * radius + cx, math.sin(a0) * radius + cy
        x1, y1 = math.cos(a1) * radius + cx, math.sin(a1) * radius + cy
        nx0, ny0 = math.cos(a0), math.sin(a0)
        nx1, ny1 = math.cos(a1), math.sin(a1)

        # Two triangles per side quad
        verts = [
            (Vec3(x0, y0, cz - half_h), Vec3(nx0, ny0, 0)),
            (Vec3(x1, y1, cz - half_h), Vec3(nx1, ny1, 0)),
            (Vec3(x1, y1, cz + half_h), Vec3(nx1, ny1, 0)),
            (Vec3(x0, y0, cz - half_h), Vec3(nx0, ny0, 0)),
            (Vec3(x1, y1, cz + half_h), Vec3(nx1, ny1, 0)),
            (Vec3(x0, y0, cz + half_h), Vec3(nx0, ny0, 0)),
        ]
        for pos, nrm in verts:
            writer_v.addData3(pos)
            writer_n.addData3(nrm)
            writer_c.addData4(color)
            writer_t.addData2(0, 0)
        prim.addVertices(idx, idx + 1, idx + 2)
        prim.addVertices(idx + 3, idx + 4, idx + 5)
        idx += 6

    # Top and bottom caps
    for z_sign, nz in [(1, Vec3(0, 0, 1)), (-1, Vec3(0, 0, -1))]:
        cz_cap = cz + half_h * z_sign
        for i in range(segments):
            a0 = 2.0 * math.pi * i / segments
            a1 = 2.0 * math.pi * (i + 1) / segments
            writer_v.addData3(cx, cy, cz_cap)
            writer_n.addData3(nz)
            writer_c.addData4(color)
            writer_t.addData2(0.5, 0.5)

            if z_sign > 0:
                writer_v.addData3(math.cos(a0) * radius + cx, math.sin(a0) * radius + cy, cz_cap)
                writer_v.addData3(math.cos(a1) * radius + cx, math.sin(a1) * radius + cy, cz_cap)
            else:
                writer_v.addData3(math.cos(a1) * radius + cx, math.sin(a1) * radius + cy, cz_cap)
                writer_v.addData3(math.cos(a0) * radius + cx, math.sin(a0) * radius + cy, cz_cap)
            for _ in range(2):
                writer_n.addData3(nz)
                writer_c.addData4(color)
                writer_t.addData2(0, 0)
            prim.addVertices(idx, idx + 1, idx + 2)
            idx += 3

    return idx


def _add_sphere_mesh(writer_v, writer_n, writer_c, writer_t, prim, idx,
                     cx, cy, cz, radius, rings, segments, color):
    """Append a UV sphere at (cx,cy,cz). Returns new idx."""
    for i in range(rings):
        theta0 = math.pi * i / rings
        theta1 = math.pi * (i + 1) / rings
        for j in range(segments):
            phi0 = 2.0 * math.pi * j / segments
            phi1 = 2.0 * math.pi * (j + 1) / segments

            def _vert(theta, phi):
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                return Vec3(x * radius + cx, y * radius + cy, z * radius + cz), Vec3(x, y, z)

            p0, n0 = _vert(theta0, phi0)
            p1, n1 = _vert(theta0, phi1)
            p2, n2 = _vert(theta1, phi1)
            p3, n3 = _vert(theta1, phi0)

            # Triangle 1
            for pos, nrm in [(p0, n0), (p1, n1), (p2, n2)]:
                writer_v.addData3(pos)
                writer_n.addData3(nrm)
                writer_c.addData4(color)
                writer_t.addData2(0, 0)
            prim.addVertices(idx, idx + 1, idx + 2)
            idx += 3

            # Triangle 2
            for pos, nrm in [(p0, n0), (p2, n2), (p3, n3)]:
                writer_v.addData3(pos)
                writer_n.addData3(nrm)
                writer_c.addData4(color)
                writer_t.addData2(0, 0)
            prim.addVertices(idx, idx + 1, idx + 2)
            idx += 3

    return idx


# -----------------------------------------------------------------------
# Standalone part builders — each returns a complete NodePath
# -----------------------------------------------------------------------

def _build_box_np(name, sx, sy, sz, color):
    """Build a box NodePath centered at origin with half-extents (sx,sy,sz)."""
    fmt = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(36)
    wv = GeomVertexWriter(vdata, 'vertex')
    wn = GeomVertexWriter(vdata, 'normal')
    wc = GeomVertexWriter(vdata, 'color')
    wt = GeomVertexWriter(vdata, 'texcoord')
    prim = GeomTriangles(Geom.UHStatic)
    _add_box(wv, wn, wc, wt, prim, 0, 0, 0, 0, sx, sy, sz, color)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    gn = GeomNode(name)
    gn.addGeom(geom)
    return NodePath(gn)


def _build_cylinder_np(name, radius, height, segments, color):
    """Build a cylinder NodePath along Z, centered at origin."""
    fmt = GeomVertexFormat.getV3n3c4t2()
    n_verts = segments * 6 + segments * 3 * 2
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(n_verts)
    wv = GeomVertexWriter(vdata, 'vertex')
    wn = GeomVertexWriter(vdata, 'normal')
    wc = GeomVertexWriter(vdata, 'color')
    wt = GeomVertexWriter(vdata, 'texcoord')
    prim = GeomTriangles(Geom.UHStatic)
    _add_cylinder(wv, wn, wc, wt, prim, 0, 0, 0, 0, radius, height, segments, color)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    gn = GeomNode(name)
    gn.addGeom(geom)
    return NodePath(gn)


def _build_sphere_np(name, radius, rings, segments, color):
    """Build a UV sphere NodePath centered at origin."""
    fmt = GeomVertexFormat.getV3n3c4t2()
    n_verts = rings * segments * 6
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(n_verts)
    wv = GeomVertexWriter(vdata, 'vertex')
    wn = GeomVertexWriter(vdata, 'normal')
    wc = GeomVertexWriter(vdata, 'color')
    wt = GeomVertexWriter(vdata, 'texcoord')
    prim = GeomTriangles(Geom.UHStatic)
    _add_sphere_mesh(wv, wn, wc, wt, prim, 0, 0, 0, 0, radius, rings, segments, color)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    gn = GeomNode(name)
    gn.addGeom(geom)
    return NodePath(gn)


# -----------------------------------------------------------------------
# Hierarchical player skeleton
# -----------------------------------------------------------------------
# CG concept — Scene-graph skeletal hierarchy:
#   The player model is assembled as a tree of NodePaths.  Each "joint"
#   is an empty NodePath whose children are geometry parts.  Rotating
#   a joint applies a rotation matrix that composes with all descendants:
#
#       M_world(foot) = M_root · M_hips · M_upper_leg · M_knee · M_foot
#
#   This is identical to how GPU-based skeletal animation works, except
#   we perform it on the CPU via the Panda3D scene graph rather than in
#   a vertex shader.  The benefit: each part can be separately colored,
#   scaled, and toggled without rebinding a skinning palette.

def make_player_mesh():
    """
    Build a hierarchical armored knight character (~2500 triangles).

    Returns a NodePath tree:
        player_root
          └── hips_joint  (z = 0.50)
              ├── spine_joint  (z = 0.35)
              │   ├── chest geometry (torso + armor plates)
              │   ├── neck_joint → head_joint → helmet + visor
              │   ├── shoulder_L_joint → upper_arm_L → elbow_L_joint → forearm_L → hand_L
              │   └── shoulder_R_joint → upper_arm_R → elbow_R_joint → forearm_R → hand_R
              ├── hip_L_joint → upper_leg_L → knee_L_joint → lower_leg_L → foot_L
              └── hip_R_joint → upper_leg_R → knee_R_joint → lower_leg_R → foot_R

    All joints are at the *pivot point* of the rotation they control.
    Geometry is offset from the joint so rotation looks natural.

    Feet at Z ≈ 0, total height ≈ 2.0 units.
    """
    segs = 10  # cylinder segments for smooth look

    # Color palette
    armor = _hex('#3d85c6')       # blue armor
    armor_dark = _hex('#2a5f9e')  # darker blue for secondary
    gold = _hex('#e8d44d')        # gold trim / accents
    visor = _hex('#4dffa6')       # green visor glow
    joint_dark = _hex('#1a1a2e')  # dark joint color
    boot_color = _hex('#2d2d3d')  # dark boots
    skin = _hex('#d4a574')        # exposed hand color

    root = NodePath('player_root')

    # ── Hips joint ── pivot at pelvis center ──────────────────────────
    hips = root.attachNewNode('hips_joint')
    hips.setPos(0, 0, 0.50)

    # Pelvis geometry (wide, short cylinder centered at joint)
    pelvis_geo = _build_cylinder_np('pelvis', 0.22, 0.18, segs, armor_dark)
    pelvis_geo.reparentTo(hips)

    # Belt/waist accent
    belt_geo = _build_cylinder_np('belt', 0.24, 0.06, segs, gold)
    belt_geo.setPos(0, 0, 0.08)
    belt_geo.reparentTo(hips)

    # ── Spine joint ── pivot at waist, above hips ─────────────────────
    spine = hips.attachNewNode('spine_joint')
    spine.setPos(0, 0, 0.12)

    # Torso: tapered cylinder (wider at chest)
    torso_geo = _build_cylinder_np('torso', 0.26, 0.50, segs, armor)
    torso_geo.setPos(0, 0, 0.28)
    torso_geo.reparentTo(spine)

    # Chest armor plate (front)
    chest_plate = _build_box_np('chest_plate', 0.20, 0.04, 0.18, armor_dark)
    chest_plate.setPos(0, 0.20, 0.35)
    chest_plate.reparentTo(spine)

    # Shoulder ridge
    shoulder_ridge = _build_cylinder_np('shoulder_ridge', 0.30, 0.08, segs, armor_dark)
    shoulder_ridge.setPos(0, 0, 0.52)
    shoulder_ridge.reparentTo(spine)

    # ── Neck joint ────────────────────────────────────────────────────
    neck = spine.attachNewNode('neck_joint')
    neck.setPos(0, 0, 0.58)

    neck_geo = _build_cylinder_np('neck', 0.08, 0.10, segs, joint_dark)
    neck_geo.setPos(0, 0, 0.05)
    neck_geo.reparentTo(neck)

    # ── Head joint ────────────────────────────────────────────────────
    head = neck.attachNewNode('head_joint')
    head.setPos(0, 0, 0.12)

    # Helmet: sphere with armor color
    helmet_geo = _build_sphere_np('helmet', 0.20, 8, 10, armor)
    helmet_geo.reparentTo(head)

    # Helmet crest (top ridge)
    crest_geo = _build_box_np('crest', 0.03, 0.12, 0.08, gold)
    crest_geo.setPos(0, 0, 0.16)
    crest_geo.reparentTo(head)

    # Visor: glowing green band across the face
    visor_geo = _build_box_np('visor', 0.16, 0.05, 0.05, visor)
    visor_geo.setPos(0, 0.17, 0.02)
    visor_geo.reparentTo(head)

    # Chin guard
    chin_geo = _build_box_np('chin_guard', 0.10, 0.06, 0.04, armor_dark)
    chin_geo.setPos(0, 0.14, -0.12)
    chin_geo.reparentTo(head)

    # ── Left shoulder joint ───────────────────────────────────────────
    shoulder_L = spine.attachNewNode('shoulder_L_joint')
    shoulder_L.setPos(-0.34, 0, 0.48)

    # Pauldron (dome-shaped shoulder armor)
    pauldron_L = _build_sphere_np('pauldron_L', 0.12, 6, 8, armor)
    pauldron_L.setScale(1, 1, 0.7)
    pauldron_L.reparentTo(shoulder_L)

    # Upper arm
    upper_arm_L = _build_cylinder_np('upper_arm_L', 0.07, 0.30, segs, armor_dark)
    upper_arm_L.setPos(0, 0, -0.18)
    upper_arm_L.reparentTo(shoulder_L)

    # Elbow joint
    elbow_L = shoulder_L.attachNewNode('elbow_L_joint')
    elbow_L.setPos(0, 0, -0.34)

    elbow_ball_L = _build_sphere_np('elbow_ball_L', 0.06, 4, 6, joint_dark)
    elbow_ball_L.reparentTo(elbow_L)

    # Forearm
    forearm_L = _build_cylinder_np('forearm_L', 0.065, 0.26, segs, armor)
    forearm_L.setPos(0, 0, -0.15)
    forearm_L.reparentTo(elbow_L)

    # Hand
    hand_L = elbow_L.attachNewNode('hand_L_joint')
    hand_L.setPos(0, 0, -0.30)
    hand_geo_L = _build_sphere_np('hand_L', 0.06, 4, 6, skin)
    hand_geo_L.reparentTo(hand_L)

    # ── Right shoulder joint ──────────────────────────────────────────
    shoulder_R = spine.attachNewNode('shoulder_R_joint')
    shoulder_R.setPos(0.34, 0, 0.48)

    pauldron_R = _build_sphere_np('pauldron_R', 0.12, 6, 8, armor)
    pauldron_R.setScale(1, 1, 0.7)
    pauldron_R.reparentTo(shoulder_R)

    upper_arm_R = _build_cylinder_np('upper_arm_R', 0.07, 0.30, segs, armor_dark)
    upper_arm_R.setPos(0, 0, -0.18)
    upper_arm_R.reparentTo(shoulder_R)

    elbow_R = shoulder_R.attachNewNode('elbow_R_joint')
    elbow_R.setPos(0, 0, -0.34)

    elbow_ball_R = _build_sphere_np('elbow_ball_R', 0.06, 4, 6, joint_dark)
    elbow_ball_R.reparentTo(elbow_R)

    forearm_R = _build_cylinder_np('forearm_R', 0.065, 0.26, segs, armor)
    forearm_R.setPos(0, 0, -0.15)
    forearm_R.reparentTo(elbow_R)

    hand_R = elbow_R.attachNewNode('hand_R_joint')
    hand_R.setPos(0, 0, -0.30)
    hand_geo_R = _build_sphere_np('hand_R', 0.06, 4, 6, skin)
    hand_geo_R.reparentTo(hand_R)

    # ── Left hip joint ────────────────────────────────────────────────
    hip_L = hips.attachNewNode('hip_L_joint')
    hip_L.setPos(-0.12, 0, -0.08)

    upper_leg_L = _build_cylinder_np('upper_leg_L', 0.09, 0.32, segs, armor_dark)
    upper_leg_L.setPos(0, 0, -0.18)
    upper_leg_L.reparentTo(hip_L)

    knee_L = hip_L.attachNewNode('knee_L_joint')
    knee_L.setPos(0, 0, -0.36)

    knee_ball_L = _build_sphere_np('knee_ball_L', 0.07, 4, 6, joint_dark)
    knee_ball_L.reparentTo(knee_L)

    # Shin + shin guard
    shin_L = _build_cylinder_np('shin_L', 0.075, 0.30, segs, armor_dark)
    shin_L.setPos(0, 0, -0.16)
    shin_L.reparentTo(knee_L)

    shin_guard_L = _build_box_np('shin_guard_L', 0.05, 0.04, 0.12, armor)
    shin_guard_L.setPos(0, 0.06, -0.14)
    shin_guard_L.reparentTo(knee_L)

    # Foot
    foot_L = knee_L.attachNewNode('foot_L_joint')
    foot_L.setPos(0, 0, -0.32)

    boot_L = _build_box_np('boot_L', 0.08, 0.12, 0.05, boot_color)
    boot_L.setPos(0, 0.02, -0.02)
    boot_L.reparentTo(foot_L)

    # ── Right hip joint ───────────────────────────────────────────────
    hip_R = hips.attachNewNode('hip_R_joint')
    hip_R.setPos(0.12, 0, -0.08)

    upper_leg_R = _build_cylinder_np('upper_leg_R', 0.09, 0.32, segs, armor_dark)
    upper_leg_R.setPos(0, 0, -0.18)
    upper_leg_R.reparentTo(hip_R)

    knee_R = hip_R.attachNewNode('knee_R_joint')
    knee_R.setPos(0, 0, -0.36)

    knee_ball_R = _build_sphere_np('knee_ball_R', 0.07, 4, 6, joint_dark)
    knee_ball_R.reparentTo(knee_R)

    shin_R = _build_cylinder_np('shin_R', 0.075, 0.30, segs, armor_dark)
    shin_R.setPos(0, 0, -0.16)
    shin_R.reparentTo(knee_R)

    shin_guard_R = _build_box_np('shin_guard_R', 0.05, 0.04, 0.12, armor)
    shin_guard_R.setPos(0, 0.06, -0.14)
    shin_guard_R.reparentTo(knee_R)

    foot_R = knee_R.attachNewNode('foot_R_joint')
    foot_R.setPos(0, 0, -0.32)

    boot_R = _build_box_np('boot_R', 0.08, 0.12, 0.05, boot_color)
    boot_R.setPos(0, 0.02, -0.02)
    boot_R.reparentTo(foot_R)

    return root


# -----------------------------------------------------------------------
# Starfield — individual billboard star quads scattered in space
# -----------------------------------------------------------------------

def make_skybox_mesh(radius=500.0, subdivisions=2, star_count=800):
    """
    Create a starfield by scattering many small billboard quads around the
    scene.  Each star is a tiny bright point at a random position on a large
    sphere, with random size and brightness.

    Also creates a dark background sphere so space isn't transparent.
    """
    from utils.math_helpers import random_point_on_sphere

    root = NodePath('starfield_root')

    # Dark background sphere (very simple, low-poly)
    bg_verts = list(_ICO_VERTS)
    bg_tris = list(_ICO_TRIS)
    bg_verts, bg_tris = _subdivide(bg_verts, bg_tris, 1)

    fmt = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData('sky_bg', fmt, Geom.UHStatic)
    vdata.setNumRows(len(bg_tris) * 3)
    wv = GeomVertexWriter(vdata, 'vertex')
    wn = GeomVertexWriter(vdata, 'normal')
    wc = GeomVertexWriter(vdata, 'color')
    wt = GeomVertexWriter(vdata, 'texcoord')
    prim = GeomTriangles(Geom.UHStatic)
    vi = 0
    for i0, i1, i2 in bg_tris:
        for idx in (i0, i2, i1):  # inward winding
            wv.addData3(bg_verts[idx] * radius)
            wn.addData3(-bg_verts[idx])
            wc.addData4(SPACE_DARK)
            wt.addData2(0, 0)
        prim.addVertices(vi, vi + 1, vi + 2)
        vi += 3
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    bg_node = GeomNode('sky_bg')
    bg_node.addGeom(geom)
    bg_np = root.attachNewNode(bg_node)
    bg_np.setLightOff()
    bg_np.setBin('background', 0)
    bg_np.setDepthWrite(False)

    # Individual star quads as billboard points
    cm = CardMaker('star')
    for _ in range(star_count):
        pos = random_point_on_sphere(radius * 0.95)
        brightness = random.uniform(0.5, 1.0)
        size = random.uniform(0.3, 1.2)

        # Occasional colored stars
        r = random.random()
        if r < 0.05:
            color = Vec4(1.0, 0.7, 0.4, brightness)  # warm
        elif r < 0.10:
            color = Vec4(0.6, 0.8, 1.0, brightness)  # cool blue
        else:
            color = Vec4(brightness, brightness, brightness, 1.0)  # white

        cm.setFrame(-size * 0.5, size * 0.5, -size * 0.5, size * 0.5)
        star = root.attachNewNode(cm.generate())
        star.setPos(pos)
        star.setColor(color)
        star.setBillboardPointEye()
        star.setLightOff()
        star.setBin('background', 1)
        star.setDepthWrite(False)

    return root


# -----------------------------------------------------------------------
# Sword weapon — proper blade shape
# -----------------------------------------------------------------------

def make_weapon_mesh():
    """
    Procedural sword with diamond cross-section blade (~130 blade triangles).

    CG concept — Diamond cross-section:
      Instead of a flat box, the blade uses a 4-point diamond profile:
        top (+Z), right (+X), bottom (-Z), left (-X)
      This produces four angled facets per segment that catch light
      differently, creating a realistic metallic sheen with flat shading.

    The blade is built from 8 tapered segments along +Z (the blade axis),
    each a frustum with diamond cross-section.  Extends along local +Z
    so it can be rotated naturally from a wrist joint.

    Returns a NodePath tree with 'sword_root' containing:
      - blade geometry
      - crossguard
      - grip
      - pommel
      - glow_overlay (slightly larger transparent blade for emissive effect)
    """
    sword_root = NodePath('sword_root')

    blade_color = _hex('#c0c8d4')   # steel silver
    edge_color = _hex('#e8eef4')    # bright edge highlights
    guard_color = _hex('#8B7500')   # gold crossguard
    grip_color = _hex('#4a3520')    # dark leather
    pommel_color = _hex('#e8d44d')  # gold pommel
    glow_color = _hex('#88bbff', 0.25)  # translucent blue glow

    # ── Grip ──────────────────────────────────────────────────────────
    grip = _build_cylinder_np('grip', 0.035, 0.32, 8, grip_color)
    # Grip along Z, centered at origin
    grip.setPos(0, 0, -0.18)
    grip.reparentTo(sword_root)

    # ── Pommel (sphere at the bottom) ─────────────────────────────────
    pommel = _build_sphere_np('pommel', 0.055, 4, 6, pommel_color)
    pommel.setPos(0, 0, -0.36)
    pommel.reparentTo(sword_root)

    # ── Crossguard ────────────────────────────────────────────────────
    guard = _build_box_np('crossguard', 0.16, 0.03, 0.045, guard_color)
    guard.setPos(0, 0, 0.0)
    guard.reparentTo(sword_root)

    # ── Diamond blade ─────────────────────────────────────────────────
    # 8 segments tapering from base to tip along +Z
    # Each cross-section is a diamond: 4 vertices at (±w, 0, z) and (0, ±d, z)
    blade_profile = [
        # (z_pos, half_width_x, half_depth_y)
        (0.04,  0.045, 0.025),
        (0.20,  0.042, 0.023),
        (0.40,  0.038, 0.020),
        (0.60,  0.033, 0.018),
        (0.80,  0.027, 0.015),
        (1.00,  0.020, 0.012),
        (1.15,  0.013, 0.008),
        (1.30,  0.006, 0.004),
        (1.42,  0.000, 0.000),  # tip
    ]

    # Build blade geometry with diamond cross-sections
    fmt = GeomVertexFormat.getV3n3c4t2()
    # Each segment between two profiles has 4 quads (8 tris), plus tip has 4 tris
    n_segs = len(blade_profile) - 1
    max_verts = n_segs * 4 * 6 + 100  # generous estimate
    vdata = GeomVertexData('blade', fmt, Geom.UHStatic)
    vdata.setNumRows(max_verts)
    wv = GeomVertexWriter(vdata, 'vertex')
    wn = GeomVertexWriter(vdata, 'normal')
    wc = GeomVertexWriter(vdata, 'color')
    wt = GeomVertexWriter(vdata, 'texcoord')
    prim = GeomTriangles(Geom.UHStatic)
    vi = 0

    def diamond_pts(z, w, d):
        """4 points of a diamond cross-section: right, front, left, back."""
        return [
            Vec3(w, 0, z),    # right
            Vec3(0, d, z),    # front
            Vec3(-w, 0, z),   # left
            Vec3(0, -d, z),   # back
        ]

    for seg in range(n_segs):
        z0, w0, d0 = blade_profile[seg]
        z1, w1, d1 = blade_profile[seg + 1]

        is_tip = (w1 < 0.001)
        pts0 = diamond_pts(z0, w0, d0)

        if is_tip:
            # Tip: 4 triangles converging to a point
            tip = Vec3(0, 0, z1)
            for fi in range(4):
                p0 = pts0[fi]
                p1 = pts0[(fi + 1) % 4]
                # Face normal via cross product
                e1 = p1 - p0
                e2 = tip - p0
                fn = normalized(e1.cross(e2))
                col = edge_color if fi % 2 == 0 else blade_color
                for p in [p0, p1, tip]:
                    wv.addData3(p)
                    wn.addData3(fn)
                    wc.addData4(col)
                    wt.addData2(0, 0)
                prim.addVertices(vi, vi + 1, vi + 2)
                vi += 3
        else:
            # Regular segment: 4 quad faces (each = 2 triangles)
            pts1 = diamond_pts(z1, w1, d1)
            for fi in range(4):
                a = pts0[fi]
                b = pts0[(fi + 1) % 4]
                c = pts1[(fi + 1) % 4]
                d = pts1[fi]
                # Flat face normal
                e1 = b - a
                e2 = d - a
                fn = normalized(e1.cross(e2))
                col = edge_color if fi % 2 == 0 else blade_color
                for p in [a, b, c]:
                    wv.addData3(p)
                    wn.addData3(fn)
                    wc.addData4(col)
                    wt.addData2(0, 0)
                prim.addVertices(vi, vi + 1, vi + 2)
                vi += 3
                for p in [a, c, d]:
                    wv.addData3(p)
                    wn.addData3(fn)
                    wc.addData4(col)
                    wt.addData2(0, 0)
                prim.addVertices(vi, vi + 1, vi + 2)
                vi += 3

    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    blade_gn = GeomNode('blade_geom')
    blade_gn.addGeom(geom)
    blade_np = NodePath(blade_gn)
    blade_np.setAttrib(ShadeModelAttrib.make(ShadeModelAttrib.MFlat))
    blade_np.reparentTo(sword_root)

    # ── Glow overlay — slightly larger translucent blade ──────────────
    # CG concept: additive blending on a semi-transparent duplicate mesh
    # creates a soft bloom/glow effect without post-processing shaders.
    glow_np = blade_np.copyTo(sword_root)
    glow_np.setName('glow_overlay')
    glow_np.setScale(1.15)
    glow_np.setColor(glow_color)
    glow_np.setTransparency(TransparencyAttrib.MAlpha)
    glow_np.setBin('transparent', 10)
    glow_np.setDepthWrite(False)

    return sword_root
