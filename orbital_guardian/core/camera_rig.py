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

from panda3d.core import Vec3, NodePath, Camera, DisplayRegion, PerspectiveLens
from utils.math_helpers import normalized, lerp_vec3, tangent_frame


class CameraRig:
    """Smooth third-person camera for spherical-planet gameplay."""

    DISTANCE = 50.0     # distance behind the player
    HEIGHT = 20.0       # height above the surface
    LOOK_HEIGHT = 1.5   # look-at point above the player
    SMOOTH = 3.5        # position interpolation speed

    # Opposite camera settings
    OPPOSITE_DISTANCE = 40.0
    OPPOSITE_HEIGHT = 15.0

    def __init__(self, camera_np, player, planet, base=None):
        self.camera = camera_np
        self.player = player
        self.planet = planet
        self.base = base  # ShowBase instance for creating display regions

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

        # Set up opposite camera (picture-in-picture)
        self._opposite_camera = None
        self._opposite_camera_np = None
        self._opposite_display_region = None
        self._opposite_orbit_dir = self._orbit_dir  # opposite direction
        self._opposite_current_pos = Vec3(0, 0, 0)
        self._opposite_current_look = Vec3(0, 0, 0)

        if self.base:
            self._setup_opposite_camera()

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

        # Update opposite camera if it exists
        if self._opposite_camera_np:
            target_opp_pos = self._ideal_opposite_position()
            target_opp_look = self._ideal_opposite_look_at()

            self._opposite_current_pos = lerp_vec3(self._opposite_current_pos, target_opp_pos, t)
            self._opposite_current_look = lerp_vec3(self._opposite_current_look, target_opp_look, t)

            self._apply_opposite()

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

    def _setup_opposite_camera(self):
        """Create a picture-in-picture camera showing the opposite side."""
        # Create a new camera node
        self._opposite_camera = Camera('opposite_camera')
        lens = PerspectiveLens()
        lens.setFov(60)
        self._opposite_camera.setLens(lens)

        # Attach to render
        self._opposite_camera_np = self.base.render.attachNewNode(self._opposite_camera)

        # Create a display region (bottom-right corner, small)
        win = self.base.win
        dr = win.makeDisplayRegion(0.72, 0.98, 0.02, 0.28)  # x1, x2, y1, y2
        dr.setCamera(self._opposite_camera_np)
        dr.setSort(20)  # Render on top
        self._opposite_display_region = dr

        # Initialize position
        self._opposite_orbit_dir = -self._orbit_dir
        self._opposite_current_pos = self._ideal_opposite_position()
        self._opposite_current_look = self._ideal_opposite_look_at()
        self._apply_opposite()

    def _ideal_opposite_position(self):
        """Position for opposite camera - on the other side of the planet."""
        pos = self.player.position
        up = self.player.get_up()

        # Opposite orbit direction (180° around planet)
        # First, get the opposite position on planet
        opposite_player_pos = self.planet.center - (pos - self.planet.center)

        # Get up at that position
        opposite_up = normalized(opposite_player_pos - self.planet.center)

        # Re-project opposite orbit direction onto tangent plane
        orb = -self._orbit_dir - opposite_up * (-self._orbit_dir).dot(opposite_up)
        if orb.length() < 1e-6:
            orb = Vec3(1, 0, 0)
        orb = normalized(orb)
        self._opposite_orbit_dir = orb

        return opposite_player_pos + opposite_up * self.OPPOSITE_HEIGHT + orb * self.OPPOSITE_DISTANCE

    def _ideal_opposite_look_at(self):
        """Look-at point for opposite camera."""
        # Look at the opposite position on planet
        opposite_player_pos = self.planet.center - (self.player.position - self.planet.center)
        opposite_up = normalized(opposite_player_pos - self.planet.center)
        return opposite_player_pos + opposite_up * self.LOOK_HEIGHT

    def _apply_opposite(self):
        """Apply transform to opposite camera."""
        if not self._opposite_camera_np:
            return

        self._opposite_camera_np.setPos(self._opposite_current_pos)

        # Get up vector at opposite position
        opposite_player_pos = self.planet.center - (self.player.position - self.planet.center)
        opposite_up = normalized(opposite_player_pos - self.planet.center)

        self._opposite_camera_np.lookAt(self._opposite_current_look, opposite_up)

    def cleanup(self):
        """Clean up display regions and camera nodes."""
        if self._opposite_display_region:
            self.base.win.removeDisplayRegion(self._opposite_display_region)
            self._opposite_display_region = None
        if self._opposite_camera_np:
            self._opposite_camera_np.removeNode()
            self._opposite_camera_np = None
