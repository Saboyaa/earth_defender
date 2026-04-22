"""
Third-person camera that orbits the player on a spherical planet.

CG concepts:
  - The camera's "up" vector is set to the player's local surface normal,
    so the horizon tilts naturally as the player walks around the sphere.
  - Smooth interpolation (lerp) prevents harsh snapping.
  - The camera maintains its own orbit direction on the tangent plane,
    independent of which way the player is facing.  This gives a stable,
    non-rotating view — the player model turns beneath a steady camera.
"""

from panda3d.core import Vec3
from utils.math_helpers import normalized, lerp_vec3, tangent_frame


class CameraRig:
    """Smooth third-person camera for spherical-planet gameplay."""

    DISTANCE = 50.0     # distance behind the player
    HEIGHT = 20.0       # height above the surface
    LOOK_HEIGHT = 1.5   # look-at point above the player
    SMOOTH = 3.5        # position interpolation speed

    def __init__(self, camera_np, player, planet):
        self.camera = camera_np
        self.player = player
        self.planet = planet

        # The camera keeps its own "orbit direction" — the direction it
        # looks FROM, on the tangent plane.  It does NOT follow the player's
        # facing, so turning the player doesn't swing the camera around.
        up = self.player.get_up()
        # Start looking from behind the player's initial forward
        self._orbit_dir = -self.player.get_forward()
        # Make sure it's on the tangent plane
        self._orbit_dir = self._orbit_dir - up * self._orbit_dir.dot(up)
        self._orbit_dir = normalized(self._orbit_dir)

        self._current_pos = self._ideal_position()
        self._current_look = self._ideal_look_at()
        self._apply()

    def _ideal_position(self):
        """Camera position: offset from player along orbit direction + up."""
        pos = self.player.position
        up = self.player.get_up()

        # Re-project orbit direction onto current tangent plane
        # (it drifts as the player walks around the sphere)
        orb = self._orbit_dir - up * self._orbit_dir.dot(up)
        if orb.length() < 1e-6:
            orb = Vec3(1, 0, 0)
        orb = normalized(orb)
        self._orbit_dir = orb

        return pos + up * self.HEIGHT + orb * self.DISTANCE

    def _ideal_look_at(self):
        """Point the camera looks at — slightly above the player."""
        up = self.player.get_up()
        return self.player.position + up * self.LOOK_HEIGHT

    def update(self, dt):
        """Called every frame to smoothly follow the player."""
        target_pos = self._ideal_position()
        target_look = self._ideal_look_at()

        t = min(1.0, self.SMOOTH * dt)
        self._current_pos = lerp_vec3(self._current_pos, target_pos, t)
        self._current_look = lerp_vec3(self._current_look, target_look, t)

        self._apply()

    def _apply(self):
        """Set the actual camera transform."""
        self.camera.setPos(self._current_pos)
        up = self.player.get_up()
        self.camera.lookAt(self._current_look, up)

    def get_forward(self):
        """
        Camera's forward direction projected onto the player's tangent plane.
        Used by the player controller so 'W' moves toward where the camera
        is looking (i.e. away from the camera).
        """
        cam_fwd = normalized(self._current_look - self._current_pos)
        up = self.player.get_up()
        projected = cam_fwd - up * cam_fwd.dot(up)
        length = projected.length()
        if length < 1e-6:
            return self.player.get_forward()
        return projected / length
