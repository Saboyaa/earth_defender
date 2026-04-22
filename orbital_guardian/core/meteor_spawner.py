"""
Meteor wave spawner with two-phase lifecycle: FALLING → EMBEDDED.

CG concepts:
  - Meteors are spawned on a large surrounding sphere and fly inward.
  - Each meteor tumbles via a per-frame rotation around a random axis.
  - On surface contact, meteors embed into the planet and start a fuse timer.
  - Embedded meteors glow with increasing intensity as the fuse burns down.
  - Wave difficulty escalates by increasing count, speed, and shortening fuses.
"""

import math
import random
from enum import Enum
from panda3d.core import Vec3, NodePath
from graphics.procedural_meshes import make_meteor_mesh
from utils.math_helpers import normalized, random_point_on_sphere, quat_from_forward_up


class MeteorState(Enum):
    FALLING = "falling"
    EMBEDDED = "embedded"


class Meteor:
    """A single meteor with a two-phase lifecycle."""

    def __init__(self, node, velocity, spin_axis, spin_speed, fuse_duration):
        self.node = node
        self.velocity = velocity
        self.spin_axis = spin_axis
        self.spin_speed = spin_speed  # degrees/sec
        self.alive = True
        self.state = MeteorState.FALLING

        # Fuse (only used when EMBEDDED)
        self.fuse_duration = fuse_duration
        self.fuse_elapsed = 0.0

        # Track spawn distance for impact marker scaling
        self.spawn_distance = node.getPos().length() if node else 90.0

    @property
    def fuse_progress(self):
        """0.0 → 1.0 as fuse burns down (1.0 = about to explode)."""
        if self.fuse_duration <= 0:
            return 1.0
        return min(1.0, self.fuse_elapsed / self.fuse_duration)

    def update(self, dt, planet_center, planet_radius, global_time):
        """
        Returns:
          'flying'   — still in the air
          'embedded' — just transitioned to embedded this frame
          'exploded' — fuse expired, deal damage
          None       — dead or cleaned up
        """
        if not self.alive:
            return None

        if self.state == MeteorState.FALLING:
            # Move toward planet
            pos = self.node.getPos()
            pos += self.velocity * dt
            self.node.setPos(pos)
            # Tumble
            self.node.setHpr(
                self.node.getH() + self.spin_axis.x * self.spin_speed * dt,
                self.node.getP() + self.spin_axis.y * self.spin_speed * dt,
                self.node.getR() + self.spin_axis.z * self.spin_speed * dt,
            )
            # Check surface contact
            dist = (pos - planet_center).length()
            if dist <= planet_radius + 0.8:
                self._embed(planet_center, planet_radius)
                return 'embedded'
            # Flew too far away?
            if dist > 150.0:
                self.destroy()
                return None
            return 'flying'

        elif self.state == MeteorState.EMBEDDED:
            self.fuse_elapsed += dt
            # Visual: ramp up glow as fuse burns
            fp = self.fuse_progress
            glow = 0.3 + 0.7 * fp
            pulse_speed = 2.0 + fp * 15.0
            pulse = 0.8 + 0.2 * math.sin(global_time * pulse_speed)
            if self.node:
                self.node.setColorScale(
                    glow * pulse,
                    glow * pulse * 0.4,
                    0.1, 1.0
                )
            # Fuse expired?
            if self.fuse_elapsed >= self.fuse_duration:
                return 'exploded'
            return 'embedded'

        return None

    def _embed(self, planet_center, planet_radius):
        """Transition from FALLING to EMBEDDED: stop movement, half-bury."""
        self.state = MeteorState.EMBEDDED
        self.velocity = Vec3(0, 0, 0)
        self.fuse_elapsed = 0.0

        if not self.node:
            return

        # Position: half-buried at the impact point
        pos = self.node.getPos()
        surface_normal = normalized(pos - planet_center)
        # Place center slightly below surface (half-buried)
        impact_pos = planet_center + surface_normal * (planet_radius - 0.3)
        self.node.setPos(impact_pos)

        # Orient: align the meteor's Z-up with the surface normal
        # Pick an arbitrary forward on the tangent plane
        arbitrary = Vec3(1, 0, 0) if abs(surface_normal.x) < 0.9 else Vec3(0, 1, 0)
        tangent = normalized(arbitrary - surface_normal * arbitrary.dot(surface_normal))
        quat = quat_from_forward_up(tangent, surface_normal)
        self.node.setQuat(quat)

    def destroy(self):
        self.alive = False
        if self.node:
            self.node.removeNode()
            self.node = None


class MeteorSpawner:
    """Manages waves of meteors attacking the planet."""

    SPAWN_RADIUS = 90.0       # distance from planet center
    BASE_SPEED = 6.0          # initial wave speed
    SPEED_INCREMENT = 0.5     # gradual speed increase per wave
    BASE_COUNT = 2            # meteors in wave 1
    COUNT_INCREMENT = 1       # +1 per wave for gentle ramp
    MAX_ACTIVE = 15
    WAVE_PAUSE = 4.0          # seconds between waves

    # Fuse timers
    BASE_FUSE = 4.0           # seconds on wave 1
    FUSE_DECREMENT = 0.3      # less time per wave
    MIN_FUSE = 1.5            # floor

    def __init__(self, planet, parent_np):
        self.planet = planet
        self.parent = parent_np
        self.meteors = []
        self.wave = 0
        self.spawned_this_wave = 0
        self.target_this_wave = 0
        self.spawn_timer = 0.0
        self.wave_pause_timer = 0.0
        self.waiting_for_wave = True
        self._global_time = 0.0
        self._start_wave()

    def _fuse_for_wave(self):
        return max(self.MIN_FUSE, self.BASE_FUSE - (self.wave - 1) * self.FUSE_DECREMENT)

    def _start_wave(self):
        self.wave += 1
        self.spawned_this_wave = 0
        self.target_this_wave = self.BASE_COUNT + (self.wave - 1) * self.COUNT_INCREMENT
        self.wave_pause_timer = self.WAVE_PAUSE
        self.waiting_for_wave = True

    def _spawn_one(self):
        """Spawn a single meteor aimed at a random surface point."""
        origin = random_point_on_sphere(self.SPAWN_RADIUS)
        target = random_point_on_sphere(self.planet.radius * 0.8)

        direction = normalized(target - origin)
        speed = self.BASE_SPEED + (self.wave - 1) * self.SPEED_INCREMENT
        velocity = direction * speed

        mesh = make_meteor_mesh(radius=random.uniform(0.6, 1.2),
                                subdivisions=1,
                                jaggedness=random.uniform(0.2, 0.45))
        mesh.reparentTo(self.parent)
        mesh.setPos(origin)

        spin_axis = Vec3(random.uniform(-1, 1),
                         random.uniform(-1, 1),
                         random.uniform(-1, 1))
        if spin_axis.length() > 0.01:
            spin_axis = normalized(spin_axis)
        else:
            spin_axis = Vec3(0, 0, 1)

        fuse = self._fuse_for_wave()
        meteor = Meteor(mesh, velocity, spin_axis,
                        spin_speed=random.uniform(60, 180),
                        fuse_duration=fuse)
        self.meteors.append(meteor)
        self.spawned_this_wave += 1

    def update(self, dt):
        """
        Advance all meteors.  Returns two lists:
          - explosions: positions where fuse expired (damage the planet)
          - newly_embedded: positions where meteors just landed (for effects)
        """
        self._global_time += dt
        explosions = []
        newly_embedded = []

        # Wave pause
        if self.waiting_for_wave:
            self.wave_pause_timer -= dt
            if self.wave_pause_timer <= 0:
                self.waiting_for_wave = False
                self.spawn_timer = 0.0
            else:
                self._update_meteors(dt, explosions, newly_embedded)
                return explosions, newly_embedded

        # Staggered spawning
        if self.spawned_this_wave < self.target_this_wave:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0:
                active_count = sum(1 for m in self.meteors if m.alive)
                if active_count < self.MAX_ACTIVE:
                    self._spawn_one()
                    self.spawn_timer = 0.6

        self._update_meteors(dt, explosions, newly_embedded)

        # Check if wave is complete
        if (self.spawned_this_wave >= self.target_this_wave
                and all(not m.alive for m in self.meteors)):
            self.meteors.clear()
            self._start_wave()

        return explosions, newly_embedded

    def _update_meteors(self, dt, explosions, newly_embedded):
        for meteor in self.meteors:
            if not meteor.alive:
                continue
            result = meteor.update(dt, self.planet.center, self.planet.radius,
                                   self._global_time)
            if result == 'embedded':
                if meteor.node:
                    newly_embedded.append(Vec3(meteor.node.getPos()))
            elif result == 'exploded':
                if meteor.node:
                    explosions.append(Vec3(meteor.node.getPos()))
                meteor.destroy()

    def get_active_meteors(self):
        """Return list of alive meteors."""
        return [m for m in self.meteors if m.alive]

    def get_falling_meteors(self):
        """Return only FALLING meteors."""
        return [m for m in self.meteors if m.alive and m.state == MeteorState.FALLING]

    def get_embedded_meteors(self):
        """Return only EMBEDDED meteors."""
        return [m for m in self.meteors if m.alive and m.state == MeteorState.EMBEDDED]

    def destroy_all(self):
        for m in self.meteors:
            m.destroy()
        self.meteors.clear()
