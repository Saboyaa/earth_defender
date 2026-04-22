"""
Lighting rig for Orbital Guardian.

CG concepts:
  - Directional light (sun): simulates a distant light source with
    parallel rays — the primary light for the scene.
  - Ambient light: low-intensity fill that prevents completely black
    shadows, tinted blue-purple for a space ambiance.
  - Point light (optional): soft warm glow at the planet core for
    subtle hemisphere fill.
"""

from panda3d.core import (
    DirectionalLight, AmbientLight, PointLight,
    Vec3, Vec4, NodePath,
)


def setup_lighting(render):
    """
    Create and attach the scene lights.  Returns a dict of light NodePaths
    so the caller can adjust them if needed.
    """
    lights = {}

    # --- Directional light (sun) ---
    sun = DirectionalLight('sun')
    sun.setColor(Vec4(1.0, 0.95, 0.8, 1.0))
    sun_np = render.attachNewNode(sun)
    # Angle from upper-right
    sun_np.setHpr(45, -45, 0)
    render.setLight(sun_np)
    lights['sun'] = sun_np

    # --- Ambient light (space fill) ---
    amb = AmbientLight('ambient')
    amb.setColor(Vec4(0.15, 0.15, 0.25, 1.0))
    amb_np = render.attachNewNode(amb)
    render.setLight(amb_np)
    lights['ambient'] = amb_np

    # --- Point light at planet center (subtle warm fill) ---
    point = PointLight('core_glow')
    point.setColor(Vec4(0.3, 0.25, 0.15, 1.0))
    point.setAttenuation(Vec3(1, 0.01, 0.001))
    point_np = render.attachNewNode(point)
    point_np.setPos(0, 0, 0)
    render.setLight(point_np)
    lights['core'] = point_np

    return lights
