"""
Meteor Impact Prediction System.

- For FALLING meteors: raycast to predict impact point, show yellow/orange marker.
- For EMBEDDED meteors: show bright red pulsing marker at the meteor's position.
- Markers grow and pulse faster as danger increases.
"""

import math
import os
from panda3d.core import (
    Vec3, Vec4, NodePath, Shader, TransparencyAttrib,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode,
)
from utils.math_helpers import normalized, quat_from_forward_up
from core.meteor_spawner import MeteorState


_DISC_SEGMENTS = 32


def _make_disc_mesh():
    """Procedural flat disc in XY plane, radius 1, UVs centered at (0.5,0.5)."""
    fmt = GeomVertexFormat.getV3t2()
    vdata = GeomVertexData('impact_disc', fmt, Geom.UHStatic)
    vdata.setNumRows(_DISC_SEGMENTS * 3)

    writer_v = GeomVertexWriter(vdata, 'vertex')
    writer_t = GeomVertexWriter(vdata, 'texcoord')
    prim = GeomTriangles(Geom.UHStatic)
    idx = 0

    for i in range(_DISC_SEGMENTS):
        a0 = 2.0 * math.pi * i / _DISC_SEGMENTS
        a1 = 2.0 * math.pi * (i + 1) / _DISC_SEGMENTS

        writer_v.addData3(0, 0, 0)
        writer_t.addData2(0.5, 0.5)

        x0, y0 = math.cos(a0), math.sin(a0)
        writer_v.addData3(x0, y0, 0)
        writer_t.addData2(0.5 + 0.5 * x0, 0.5 + 0.5 * y0)

        x1, y1 = math.cos(a1), math.sin(a1)
        writer_v.addData3(x1, y1, 0)
        writer_t.addData2(0.5 + 0.5 * x1, 0.5 + 0.5 * y1)

        prim.addVertices(idx, idx + 1, idx + 2)
        idx += 3

    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('disc')
    node.addGeom(geom)
    return node


def _ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    """Classic analytic ray-sphere intersection. Returns hit point or None."""
    oc = ray_origin - sphere_center
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    t = t1 if t1 > 0 else t2
    if t < 0:
        return None
    return ray_origin + ray_dir * t


class ImpactMarker:
    """A single target marker on the planet surface."""

    def __init__(self, disc_geom_node, parent_np, shader):
        self.node = parent_np.attachNewNode('marker_root')
        self.node.attachNewNode(disc_geom_node.makeCopy())
        self.node.setShader(shader)
        self.node.setTransparency(TransparencyAttrib.MAlpha)
        self.node.setBin('transparent', 5)
        self.node.setDepthWrite(False)
        self.node.setLightOff()
        self.node.setShaderInput('time', 0.0)
        self.node.setShaderInput('urgency', 0.0)
        self.node.setShaderInput('opacity', 0.0)
        self.node.setShaderInput('markerColor', Vec3(1, 0.8, 0))
        self.active = False

    def place(self, surface_point, surface_normal, radius, urgency, time_val,
              color=None):
        """Position and orient the marker on the planet surface."""
        self.active = True
        pos = surface_point + surface_normal * 0.05
        self.node.setPos(pos)

        arbitrary = Vec3(1, 0, 0) if abs(surface_normal.x) < 0.9 else Vec3(0, 1, 0)
        tangent = normalized(arbitrary - surface_normal * arbitrary.dot(surface_normal))
        quat = quat_from_forward_up(tangent, surface_normal)
        self.node.setQuat(quat)
        self.node.setScale(radius)

        self.node.setShaderInput('time', time_val)
        self.node.setShaderInput('urgency', urgency)
        self.node.setShaderInput('opacity', min(1.0, 0.3 + urgency * 0.7))
        if color:
            self.node.setShaderInput('markerColor', color)
        self.node.show()

    def hide(self):
        self.active = False
        self.node.hide()

    def destroy(self):
        self.node.removeNode()


# Marker colors
COLOR_FALLING = Vec3(1.0, 0.75, 0.0)    # yellow-orange
COLOR_EMBEDDED = Vec3(1.0, 0.15, 0.05)  # bright red


class ImpactPredictor:
    """Manages impact prediction markers for all active meteors."""

    SPAWN_RADIUS = 90.0
    MIN_MARKER_RADIUS = 0.4
    MAX_MARKER_RADIUS = 2.0

    def __init__(self, planet, parent_np):
        self.planet = planet
        self.parent = parent_np
        self.markers = []
        self._disc_geom = _make_disc_mesh()

        shader_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'graphics', 'shaders'
        )
        self._shader = Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(shader_dir, 'impact_marker.vert.glsl'),
            fragment=os.path.join(shader_dir, 'impact_marker.frag.glsl'),
        )

    def _get_marker(self, index):
        while index >= len(self.markers):
            self.markers.append(
                ImpactMarker(self._disc_geom, self.parent, self._shader)
            )
        return self.markers[index]

    def update(self, active_meteors, time_val):
        """Update markers for all active meteors."""
        used = 0

        for meteor in active_meteors:
            if not meteor.alive or not meteor.node:
                continue

            meteor_pos = meteor.node.getPos()

            if meteor.state == MeteorState.FALLING:
                # Raycast to predict impact point
                if meteor.velocity.length() < 0.01:
                    continue
                ray_dir = normalized(meteor.velocity)
                hit = _ray_sphere_intersect(
                    meteor_pos, ray_dir,
                    self.planet.center, self.planet.radius
                )
                if hit is None:
                    continue

                surface_normal = normalized(hit - self.planet.center)
                dist_to_surface = (meteor_pos - self.planet.center).length() - self.planet.radius
                max_dist = self.SPAWN_RADIUS - self.planet.radius
                closeness = 1.0 - max(0.0, min(1.0, dist_to_surface / max_dist))

                radius = self.MIN_MARKER_RADIUS + closeness * (self.MAX_MARKER_RADIUS - self.MIN_MARKER_RADIUS)

                marker = self._get_marker(used)
                marker.place(hit, surface_normal, radius, closeness, time_val,
                             color=COLOR_FALLING)
                used += 1

            elif meteor.state == MeteorState.EMBEDDED:
                # Marker at the meteor's actual position on the surface
                surface_normal = normalized(meteor_pos - self.planet.center)
                surface_point = self.planet.center + surface_normal * self.planet.radius

                urgency = meteor.fuse_progress  # 0→1 as fuse burns
                radius = self.MAX_MARKER_RADIUS * (0.8 + 0.2 * urgency)

                marker = self._get_marker(used)
                marker.place(surface_point, surface_normal, radius, urgency,
                             time_val, color=COLOR_EMBEDDED)
                used += 1

        # Hide unused markers
        for i in range(used, len(self.markers)):
            self.markers[i].hide()

    def cleanup(self):
        for m in self.markers:
            m.destroy()
        self.markers.clear()
