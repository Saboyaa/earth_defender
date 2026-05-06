"""
Health pickup system - spawns healing items on the planet surface.

The player can collect health pickups by walking over them to restore
planet health. Pickups spawn at random intervals on the planet surface.
"""

import random
from panda3d.core import Vec3, NodePath, GeomNode
from graphics.procedural_meshes import make_meteor_mesh
from utils.math_helpers import normalized, random_point_on_sphere, quat_from_forward_up


class HealthPickup:
    """A single health pickup item on the planet surface."""

    def __init__(self, node, position, heal_amount=25.0):
        self.node = node
        self.position = position
        self.heal_amount = heal_amount
        self.alive = True
        self.bob_time = 0.0  # for animated bobbing

    def update(self, dt):
        """Animate the pickup with bobbing and rotation."""
        if not self.alive or not self.node:
            return

        self.bob_time += dt
        # Gentle bobbing up and down
        import math
        bob_offset = math.sin(self.bob_time * 2.0) * 0.3

        # Rotate for visibility
        self.node.setH(self.node.getH() + 100 * dt)

        # More vibrant green color with stronger pulse and glow
        pulse = 0.5 + 0.5 * math.sin(self.bob_time * 4.0)
        glow_intensity = 1.5 + 0.8 * math.sin(self.bob_time * 3.0)

        # Bright neon green with glow
        self.node.setColorScale(
            0.3 * glow_intensity,      # R - minimal red
            2.2 * glow_intensity,      # G - intense green (>1 for glow effect)
            0.5 * glow_intensity,      # B - some cyan
            1.0
        )

    def collect(self):
        """Mark pickup as collected and remove from scene."""
        self.alive = False
        if self.node:
            self.node.removeNode()
            self.node = None

    def destroy(self):
        """Clean up the pickup."""
        self.collect()


class HealthPickupSpawner:
    """Manages spawning and collection of health pickups."""

    SPAWN_INTERVAL_MIN = 8.0   # minimum seconds between spawns
    SPAWN_INTERVAL_MAX = 15.0  # maximum seconds between spawns
    PICKUP_RADIUS = 2.0        # collection radius
    MAX_PICKUPS = 3            # max simultaneous pickups on map
    HEAL_AMOUNT = 30.0         # health restored per pickup

    def __init__(self, planet, player, parent_np):
        self.planet = planet
        self.player = player
        self.parent = parent_np
        self.pickups = []
        self.spawn_timer = random.uniform(self.SPAWN_INTERVAL_MIN,
                                         self.SPAWN_INTERVAL_MAX)

    def _spawn_pickup(self):
        """Spawn a health pickup at a random location on the planet surface."""
        # Don't spawn if at max capacity
        active_count = sum(1 for p in self.pickups if p.alive)
        if active_count >= self.MAX_PICKUPS:
            return

        # Random position on planet surface
        pos = random_point_on_sphere(self.planet.radius * 0.9)
        surface_normal = normalized(pos - self.planet.center)
        surface_pos = self.planet.center + surface_normal * (self.planet.radius + 0.5)

        # Create a green glowing crystal mesh (larger and brighter)
        mesh = make_meteor_mesh(radius=0.8, subdivisions=1, jaggedness=0.25)
        mesh.reparentTo(self.parent)
        mesh.setPos(surface_pos)

        # Orient it to stand on the surface
        arbitrary = Vec3(1, 0, 0) if abs(surface_normal.x) < 0.9 else Vec3(0, 1, 0)
        tangent = normalized(arbitrary - surface_normal * arbitrary.dot(surface_normal))
        quat = quat_from_forward_up(tangent, surface_normal)
        mesh.setQuat(quat)

        # Bright neon green healing color with glow
        mesh.setColorScale(0.3, 2.2, 0.5, 1.0)

        pickup = HealthPickup(mesh, surface_pos, self.HEAL_AMOUNT)
        self.pickups.append(pickup)

    def update(self, dt):
        """
        Update spawn timer and check for pickups collected by player.
        Returns amount of health to restore (0 if none collected).
        """
        health_restored = 0.0

        # Update spawn timer
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self._spawn_pickup()
            self.spawn_timer = random.uniform(self.SPAWN_INTERVAL_MIN,
                                             self.SPAWN_INTERVAL_MAX)

        # Update all pickups
        for pickup in self.pickups:
            if not pickup.alive:
                continue

            pickup.update(dt)

            # Check if player is close enough to collect
            dist = (pickup.position - self.player.position).length()
            if dist < self.PICKUP_RADIUS:
                health_restored += pickup.heal_amount
                pickup.collect()

        # Clean up dead pickups
        self.pickups = [p for p in self.pickups if p.alive]

        return health_restored

    def cleanup(self):
        """Remove all pickups from the scene."""
        for pickup in self.pickups:
            pickup.destroy()
        self.pickups.clear()
