"""
Heads-Up Display — planet health bar, score, wave counter, cooldown indicator,
and directional arrows pointing toward off-screen meteors.

CG concepts:
  - 3D-to-screen-space projection for off-screen indicator arrows.
  - Procedural arrow geometry using CardMaker quads with rotation.
"""

import math
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    CardMaker, NodePath, Vec3, Vec4, Point3, Point2,
    TransparencyAttrib, TextNode, LMatrix4f,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode,
)


class HUD:
    """Game HUD drawn in Panda3D's aspect2d overlay."""

    MAX_ARROWS = 20
    ARROW_MARGIN = 0.08  # distance from screen edge
    ARROW_SCALE = 0.04

    def __init__(self, aspect2d):
        self.root = aspect2d
        self._elements = []

        # --- Planet health bar (top center) ---
        self._health_bg = self._make_bar(
            self.root, pos=(0, 0, 0.92), sx=0.6, sy=0.03,
            color=Vec4(0.3, 0.0, 0.0, 0.8)
        )
        self._health_fg = self._make_bar(
            self.root, pos=(0, 0, 0.92), sx=0.6, sy=0.03,
            color=Vec4(1.0, 0.25, 0.25, 0.9)
        )
        self._health_label = OnscreenText(
            text='Planet Health', pos=(0, 0.96), scale=0.045,
            fg=(1, 1, 1, 0.9), shadow=(0, 0, 0, 0.5),
            parent=self.root, mayChange=True
        )
        self._elements.extend([self._health_bg, self._health_fg, self._health_label])

        # --- Score (top right) ---
        self._score_text = OnscreenText(
            text='Score: 0', pos=(1.2, 0.92), scale=0.055,
            fg=(0.27, 1.0, 0.53, 1.0), shadow=(0, 0, 0, 0.5),
            align=TextNode.ARight, parent=self.root, mayChange=True
        )
        self._elements.append(self._score_text)

        # --- Wave counter (top left) ---
        self._wave_text = OnscreenText(
            text='Wave 1', pos=(-1.2, 0.92), scale=0.055,
            fg=(1, 1, 1, 0.9), shadow=(0, 0, 0, 0.5),
            align=TextNode.ALeft, parent=self.root, mayChange=True
        )
        self._elements.append(self._wave_text)

        # --- Cooldown indicator ---
        self._cd_bg = self._make_bar(
            self.root, pos=(0, 0, 0.85), sx=0.15, sy=0.015,
            color=Vec4(0.2, 0.2, 0.2, 0.6)
        )
        self._cd_fg = self._make_bar(
            self.root, pos=(0, 0, 0.85), sx=0.15, sy=0.015,
            color=Vec4(0.5, 0.7, 1.0, 0.8)
        )
        self._cd_label = OnscreenText(
            text='ATK', pos=(0, 0.865), scale=0.03,
            fg=(0.7, 0.85, 1.0, 0.8), shadow=(0, 0, 0, 0.3),
            parent=self.root, mayChange=True
        )
        self._elements.extend([self._cd_bg, self._cd_fg, self._cd_label])

        # --- Directional arrows (pre-allocated pool) ---
        self._arrows = []
        for _ in range(self.MAX_ARROWS):
            arrow = self._make_arrow_node()
            arrow.hide()
            self._arrows.append(arrow)
            self._elements.append(arrow)

    @staticmethod
    def _make_bar(parent, pos, sx, sy, color):
        cm = CardMaker('bar')
        cm.setFrame(-1, 1, -1, 1)
        np = parent.attachNewNode(cm.generate())
        np.setPos(pos[0], pos[1], pos[2])
        np.setScale(sx, 1, sy)
        np.setColor(color)
        np.setTransparency(TransparencyAttrib.MAlpha)
        return np

    def _make_arrow_node(self):
        """
        Create a chevron arrow shape using procedural geometry.
        Points right (+X in screen space), rotated via setR() toward target.
        Shape: a bold chevron ► with a pointed tip and angled sides.
        """
        fmt = GeomVertexFormat.getV3()
        vdata = GeomVertexData('arrow', fmt, Geom.UHStatic)
        vdata.setNumRows(9)
        writer = GeomVertexWriter(vdata, 'vertex')

        # Chevron pointing right: outer triangle + inner cutout = 3 triangles
        # Tip at (1.5, 0), top at (-0.5, 0.8), bottom at (-0.5, -0.8)
        # Inner notch at (0.2, 0)
        #
        #        *  (tip)
        #       / \
        #      /   \
        #     / .-' \
        #    /.'     \
        #   *----*----*
        #  top  notch bottom

        # Triangle 1: top-left to tip to notch
        writer.addData3(-0.5, 0, 0.8)    # top-left
        writer.addData3(1.5, 0, 0.0)     # tip
        writer.addData3(0.2, 0, 0.0)     # inner notch

        # Triangle 2: tip to bottom-left to notch
        writer.addData3(1.5, 0, 0.0)     # tip
        writer.addData3(-0.5, 0, -0.8)   # bottom-left
        writer.addData3(0.2, 0, 0.0)     # inner notch

        # Triangle 3: top-left to notch to bottom-left
        writer.addData3(-0.5, 0, 0.8)    # top-left
        writer.addData3(0.2, 0, 0.0)     # inner notch
        writer.addData3(-0.5, 0, -0.8)   # bottom-left

        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(3, 4, 5)
        prim.addVertices(6, 7, 8)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node = GeomNode('arrow_geom')
        geom_node.addGeom(geom)

        np = self.root.attachNewNode(geom_node)
        np.setScale(self.ARROW_SCALE)
        np.setTransparency(TransparencyAttrib.MAlpha)
        return np

    def update(self, health_frac, score, wave, cooldown_frac):
        # Health bar
        self._health_fg.setScale(0.6 * max(0.0, health_frac), 1, 0.03)
        self._health_fg.setX(-0.6 * (1.0 - max(0.0, health_frac)))

        self._score_text.setText(f'Score: {score}')
        self._wave_text.setText(f'Wave {wave}')

        # Cooldown bar
        ready = 1.0 - cooldown_frac
        self._cd_fg.setScale(0.15 * ready, 1, 0.015)
        self._cd_fg.setX(-0.15 * (1.0 - ready))

    def update_arrows(self, meteor_screen_data):
        """
        Update directional arrows for off-screen meteors.

        *meteor_screen_data*: list of (screen_x, screen_y, is_on_screen, is_embedded, distance)
        where screen_x/y are in aspect2d coordinates (-1.33..1.33, -1..1).
        """
        used = 0

        for sx, sy, on_screen, is_embedded, dist in meteor_screen_data:
            if on_screen:
                continue  # visible on screen, no arrow needed
            if used >= self.MAX_ARROWS:
                break

            arrow = self._arrows[used]
            used += 1

            # Color: red for embedded, yellow for falling
            if is_embedded:
                arrow.setColor(1.0, 0.2, 0.1, 0.9)
            else:
                arrow.setColor(1.0, 0.8, 0.0, 0.7)

            # Compute direction from screen center to the off-screen point
            angle = math.atan2(sy, sx)

            # Clamp to screen edge with margin
            # Aspect ratio ~1.33 for 4:3 or wider
            edge_x = 1.25 - self.ARROW_MARGIN
            edge_y = 0.95 - self.ARROW_MARGIN

            # Find where the ray from center hits the screen edge
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            if abs(cos_a) < 0.001:
                t = edge_y / abs(sin_a)
            elif abs(sin_a) < 0.001:
                t = edge_x / abs(cos_a)
            else:
                tx = edge_x / abs(cos_a)
                ty = edge_y / abs(sin_a)
                t = min(tx, ty)

            ax = cos_a * t
            ay = sin_a * t

            arrow.setPos(ax, 0, ay)
            # Rotate arrow to point toward the meteor direction
            arrow.setR(-math.degrees(angle))

            # Scale with distance (closer = bigger)
            scale = self.ARROW_SCALE * max(0.5, min(1.5, 80.0 / max(dist, 1.0)))
            arrow.setScale(scale)
            arrow.show()

        # Hide unused arrows
        for i in range(used, self.MAX_ARROWS):
            self._arrows[i].hide()

    def show_message(self, text, scale=0.1):
        self._center_msg = OnscreenText(
            text=text, pos=(0, 0), scale=scale,
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.7),
            parent=self.root, mayChange=True
        )
        self._elements.append(self._center_msg)
        return self._center_msg

    def hide(self):
        for e in self._elements:
            if isinstance(e, NodePath):
                e.hide()
            elif isinstance(e, OnscreenText):
                e.hide()

    def show(self):
        for e in self._elements:
            if isinstance(e, NodePath):
                e.show()
            elif isinstance(e, OnscreenText):
                e.show()

    def cleanup(self):
        for e in self._elements:
            if isinstance(e, NodePath):
                e.removeNode()
            elif isinstance(e, OnscreenText):
                e.destroy()
        self._elements.clear()
        self._arrows.clear()
