"""
Planet — spherical gravity field and surface management.

CG concepts:
  - Spherical gravity: all dynamic entities are pulled toward the planet
    center by a radial force field.
  - Surface snapping: after physics integration, entities are re-projected
    onto the sphere surface to prevent sinking/floating.
  - Local 'up' vector: the outward-pointing surface normal at any point
    equals the normalised offset from the planet center.
"""

from panda3d.core import Vec3, NodePath, TransparencyAttrib
from graphics.procedural_meshes import make_planet_mesh, make_atmosphere_mesh
from utils.math_helpers import normalized, snap_to_surface, surface_up


class Planet:
    """Manages the planet mesh, atmosphere shell, and gravity field."""

    RADIUS = 10.0
    GRAVITY_STRENGTH = 25.0  # units/s²

    def __init__(self, parent_np):
        self.center = Vec3(0, 0, 0)
        self.radius = self.RADIUS
        self.health = 100.0
        self.max_health = 100.0

        # Generate planet mesh
        self.node = make_planet_mesh(radius=self.radius, subdivisions=3,
                                     noise_scale=1.5, noise_amplitude=0.4)
        self.node.reparentTo(parent_np)

        # Atmosphere disabled
        self.atmosphere = NodePath('empty_atmosphere')

    # ------------------------------------------------------------------
    # Gravity API
    # ------------------------------------------------------------------

    def gravity_direction(self, pos):
        """Unit vector pointing from *pos* toward the planet center."""
        return normalized(self.center - pos)

    def apply_gravity(self, pos, velocity, dt):
        """
        Apply gravitational acceleration to *velocity* and return the
        updated velocity.  Does NOT move the entity — the caller is
        responsible for integrating position.
        """
        g_dir = self.gravity_direction(pos)
        velocity += g_dir * self.GRAVITY_STRENGTH * dt
        return velocity

    def snap_to_surface(self, pos, height_offset=0.0):
        """Re-project *pos* onto the sphere surface at radius + offset."""
        return snap_to_surface(pos, self.radius, height_offset, self.center)

    def local_up(self, pos):
        """Surface normal (outward) at the closest point on the sphere."""
        return surface_up(pos, self.center)

    def distance_to_surface(self, pos):
        """Signed distance from *pos* to the planet surface (positive = above)."""
        return (pos - self.center).length() - self.radius

    # ------------------------------------------------------------------
    # Gameplay
    # ------------------------------------------------------------------

    def take_damage(self, amount):
        """Reduce planet health.  Returns True if the planet is destroyed."""
        self.health = max(0.0, self.health - amount)
        return self.health <= 0.0
