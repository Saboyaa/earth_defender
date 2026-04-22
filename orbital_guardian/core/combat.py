"""
Melee combat system with two-phase meteor targeting.

- Hitting a FALLING meteor awards 3x bonus points (skillful play).
- Hitting an EMBEDDED meteor awards base points scaled by fuse progress
  (more points for last-second saves).
- Auto-aim: the hitbox center nudges toward the nearest embedded meteor
  within range, making it forgiving to land hits.
- The player's sword swing animation is triggered on attack.
"""

from panda3d.core import Vec3
from utils.math_helpers import normalized
from core.meteor_spawner import MeteorState


class CombatSystem:
    """Manages melee attacks, cooldowns, and meteor hit detection."""

    ATTACK_DURATION = 0.47   # seconds the swing is active (matches player animation)
    COOLDOWN = 0.4           # seconds between attacks
    HIT_RADIUS = 3.0         # radius of the attack hitbox
    HIT_DISTANCE = 3.0       # base hitbox center distance ahead of player
    AUTO_AIM_RANGE = 5.0     # max distance to auto-aim toward embedded meteor
    BASE_SCORE = 100
    FALLING_MULTIPLIER = 3   # bonus for hitting in-flight meteors

    def __init__(self, player, meteor_spawner):
        self.player = player
        self.spawner = meteor_spawner
        self.attacking = False
        self.attack_timer = 0.0
        self.cooldown_timer = 0.0
        self.score = 0
        self._hits_this_frame = []       # (position, was_embedded) tuples
        self._safe_destroys = []          # positions of punched embedded meteors (green fx)
        self._falling_destroys = []       # positions of punched falling meteors (blue fx)

    def try_attack(self):
        """Initiate an attack if cooldown has elapsed."""
        if self.cooldown_timer > 0 or self.attacking:
            return False
        self.attacking = True
        self.attack_timer = self.ATTACK_DURATION
        # Trigger sword swing on the player
        self.player.start_swing()
        return True

    def update(self, dt):
        """Advance timers, check for hits during the active swing window."""
        self._hits_this_frame.clear()
        self._safe_destroys.clear()
        self._falling_destroys.clear()

        if self.cooldown_timer > 0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

        if not self.attacking:
            return

        self.attack_timer -= dt
        if self.attack_timer <= 0:
            self.attacking = False
            self.cooldown_timer = self.COOLDOWN
            return

        # Compute hitbox center — with auto-aim nudge
        base_center = (self.player.position
                       + self.player.get_forward() * self.HIT_DISTANCE
                       + self.player.get_up() * 1.0)

        hitbox_center = self._auto_aim(base_center)

        # Test against all alive meteors
        for meteor in self.spawner.get_active_meteors():
            if not meteor.alive or not meteor.node:
                continue
            meteor_pos = meteor.node.getPos()
            dist = (meteor_pos - hitbox_center).length()

            # Generous hitbox for embedded, tighter for falling
            effective_radius = self.HIT_RADIUS
            if meteor.state == MeteorState.EMBEDDED:
                effective_radius = self.HIT_RADIUS + 0.5  # extra generous
            meteor_radius = 1.2

            if dist < effective_radius + meteor_radius:
                if meteor.state == MeteorState.EMBEDDED:
                    # Score scales with fuse progress — more for last-second
                    points = int(self.BASE_SCORE * (1.0 + meteor.fuse_progress))
                    self.score += points
                    self._safe_destroys.append(Vec3(meteor_pos))
                elif meteor.state == MeteorState.FALLING:
                    # Bonus for hitting in-flight
                    self.score += self.BASE_SCORE * self.FALLING_MULTIPLIER
                    self._falling_destroys.append(Vec3(meteor_pos))

                self._hits_this_frame.append(Vec3(meteor_pos))
                meteor.destroy()

    def _auto_aim(self, base_center):
        """
        Nudge the hitbox center toward the nearest embedded meteor if one
        is within AUTO_AIM_RANGE.  Makes it forgiving to punch landed rocks.
        """
        nearest = None
        nearest_dist = self.AUTO_AIM_RANGE

        for meteor in self.spawner.get_embedded_meteors():
            if not meteor.node:
                continue
            dist = (meteor.node.getPos() - base_center).length()
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = meteor

        if nearest and nearest.node:
            # Nudge 40% toward the nearest embedded meteor
            target = nearest.node.getPos()
            return base_center + (target - base_center) * 0.4

        return base_center

    def get_hits(self):
        """All positions where meteors were destroyed this frame."""
        return list(self._hits_this_frame)

    def get_safe_destroys(self):
        """Positions of punched EMBEDDED meteors (green/blue particles)."""
        return list(self._safe_destroys)

    def get_falling_destroys(self):
        """Positions of punched FALLING meteors (bonus fx)."""
        return list(self._falling_destroys)

    def get_attack_progress(self):
        if not self.attacking:
            return 0.0
        return 1.0 - (self.attack_timer / self.ATTACK_DURATION)

    def get_cooldown_fraction(self):
        if self.COOLDOWN <= 0:
            return 0.0
        return self.cooldown_timer / self.COOLDOWN
