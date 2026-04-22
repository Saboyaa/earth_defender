"""
Particle effects — explosions, meteor trails, and attack swoosh.

CG concepts:
  - Billboard particles: small quads that always face the camera, giving
    the illusion of volumetric effects with minimal geometry.
  - Alpha fade-out over lifetime for smooth dissipation.
  - Colour interpolation from hot to cool over the particle's life.

Implementation uses manually managed NodePaths with CardMaker quads
rather than Panda3D's built-in particle system, to keep things explicit
and demonstrate the underlying CG principles.
"""

import random
from panda3d.core import (
    Vec3, Vec4, CardMaker, NodePath, TransparencyAttrib, BillboardEffect,
)
from utils.math_helpers import normalized


class Particle:
    """A single billboard particle with position, velocity, colour, and lifetime."""
    __slots__ = ('node', 'velocity', 'life', 'max_life', 'start_color', 'end_color')

    def __init__(self, node, velocity, life, start_color, end_color):
        self.node = node
        self.velocity = velocity
        self.life = life
        self.max_life = life
        self.start_color = start_color
        self.end_color = end_color


class ParticleManager:
    """Manages all active particles in the scene."""

    def __init__(self, parent_np):
        self.parent = parent_np
        self.particles = []
        # Reusable card geometry
        self._card = CardMaker('particle')
        self._card.setFrame(-0.15, 0.15, -0.15, 0.15)

    def spawn_explosion(self, position, count=25):
        """Burst of orange/yellow particles expanding outward — meteor destruction."""
        for _ in range(count):
            direction = Vec3(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            )
            if direction.length() < 0.01:
                direction = Vec3(0, 0, 1)
            direction = normalized(direction)
            speed = random.uniform(4.0, 12.0)

            node = self.parent.attachNewNode(self._card.generate())
            node.setPos(position)
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setBillboardPointEye()
            node.setLightOff()
            node.setBin('fixed', 40)
            node.setDepthWrite(False)
            scale = random.uniform(0.2, 0.5)
            node.setScale(scale)

            life = random.uniform(0.3, 0.7)
            start_c = Vec4(1.0, random.uniform(0.5, 0.9), 0.0, 1.0)
            end_c = Vec4(0.8, 0.2, 0.0, 0.0)

            self.particles.append(
                Particle(node, direction * speed, life, start_c, end_c)
            )

    def spawn_trail(self, position, velocity_hint):
        """Single trail particle behind a moving meteor."""
        node = self.parent.attachNewNode(self._card.generate())
        node.setPos(position)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setBillboardPointEye()
        node.setLightOff()
        node.setBin('fixed', 40)
        node.setDepthWrite(False)
        node.setScale(random.uniform(0.15, 0.3))

        # Trail drifts slightly opposite to meteor direction
        drift = Vec3(
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5),
        )
        start_c = Vec4(1.0, 0.6, 0.1, 0.7)
        end_c = Vec4(0.5, 0.1, 0.0, 0.0)
        self.particles.append(
            Particle(node, drift, random.uniform(0.3, 0.6), start_c, end_c)
        )

    def spawn_attack_swoosh(self, position, direction, count=10):
        """Arc of blue-white particles along the swing path."""
        right = Vec3(-direction.y, direction.x, 0)
        if right.length() < 0.01:
            right = Vec3(1, 0, 0)
        right = normalized(right)

        for i in range(count):
            t = i / max(count - 1, 1)
            offset = direction * (1.0 + t * 2.0) + right * (t - 0.5) * 2.0
            node = self.parent.attachNewNode(self._card.generate())
            node.setPos(position + offset)
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setBillboardPointEye()
            node.setLightOff()
            node.setBin('fixed', 40)
            node.setDepthWrite(False)
            node.setScale(random.uniform(0.1, 0.25))

            vel = offset * 0.5
            start_c = Vec4(0.7, 0.85, 1.0, 0.9)
            end_c = Vec4(0.3, 0.5, 1.0, 0.0)
            self.particles.append(
                Particle(node, vel, random.uniform(0.2, 0.4), start_c, end_c)
            )

    def spawn_impact(self, position, count=15):
        """Ground-hit particles when a meteor strikes the planet."""
        for _ in range(count):
            direction = normalized(position)  # outward from planet center
            spread = Vec3(
                direction.x + random.uniform(-0.5, 0.5),
                direction.y + random.uniform(-0.5, 0.5),
                direction.z + random.uniform(-0.5, 0.5),
            )
            speed = random.uniform(3.0, 8.0)

            node = self.parent.attachNewNode(self._card.generate())
            node.setPos(position)
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setBillboardPointEye()
            node.setLightOff()
            node.setBin('fixed', 40)
            node.setDepthWrite(False)
            node.setScale(random.uniform(0.2, 0.4))

            life = random.uniform(0.4, 0.8)
            start_c = Vec4(1.0, 0.3, 0.1, 1.0)
            end_c = Vec4(0.4, 0.1, 0.0, 0.0)
            self.particles.append(
                Particle(node, normalized(spread) * speed, life, start_c, end_c)
            )

    def update(self, dt):
        """Advance all particles; remove dead ones."""
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                p.node.removeNode()
                continue
            # Move
            pos = p.node.getPos()
            pos += p.velocity * dt
            p.node.setPos(pos)
            # Interpolate colour
            t = 1.0 - (p.life / p.max_life)
            c = Vec4(
                p.start_color.x + t * (p.end_color.x - p.start_color.x),
                p.start_color.y + t * (p.end_color.y - p.start_color.y),
                p.start_color.z + t * (p.end_color.z - p.start_color.z),
                p.start_color.w + t * (p.end_color.w - p.start_color.w),
            )
            p.node.setColor(c)
            alive.append(p)
        self.particles = alive

    def cleanup(self):
        for p in self.particles:
            p.node.removeNode()
        self.particles.clear()
