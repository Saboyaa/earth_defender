"""
Player controller with hierarchical skeleton and procedural animation.

CG concepts demonstrated:
  - **Affine transformation hierarchy**: the character is a tree of joints
    (NodePaths).  Rotating a parent joint composes its 4×4 matrix with every
    descendant, exactly like GPU skeletal animation but executed by the
    scene graph on the CPU.  M_world = M_root · M_hips · M_thigh · …
  - **Quaternion orientation**: the player's world-space facing uses a
    quaternion built from the local {right, forward, up} frame, avoiding
    the gimbal-lock that Euler angles would cause near the poles.
  - **Tangent-plane movement**: WASD input is projected onto the sphere's
    tangent plane at the player's feet, then re-snapped to the surface.
  - **Procedural animation**: all motion is computed analytically per-frame
    using sin/cos wave functions and easing curves — no keyframed data.
  - **Animation blending**: transitions between states use lerp on joint
    angles so poses flow smoothly instead of popping.
  - **Sword trail**: a dynamic quad-strip mesh records recent blade-tip
    positions and fades over time, demonstrating real-time mesh updates.
"""

import math
from enum import Enum, auto
from panda3d.core import (
    Vec3, Vec4, LQuaternionf, NodePath, TransparencyAttrib,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTristrips, GeomNode,
)
from graphics.procedural_meshes import make_player_mesh, make_weapon_mesh
from utils.math_helpers import (
    normalized, tangent_frame, quat_from_forward_up, slerp_quat,
    snap_to_surface, lerp_vec3, slerp_vec3, lerp_float,
    ease_in_quad, ease_out_cubic, ease_in_out_quad, ease_out_elastic,
)


# ======================================================================
# Animation state machine
# ======================================================================

class AnimState(Enum):
    """
    Discrete animation states.  Each state defines a set of target joint
    angles that are blended toward every frame.  Transitions happen via
    lerp so there is never a hard pop between poses.
    """
    IDLE = auto()
    WALKING = auto()
    ATTACKING = auto()
    HIT_REACT = auto()


# ======================================================================
# Joint angle poses — dict of {joint_name: (H, P, R)} in degrees
# ======================================================================
# CG concept: a "pose" is a snapshot of all joint rotations.  By linearly
# interpolating (lerp) between two poses, we get smooth transitions.
# Each joint stores heading (H), pitch (P), and roll (R) — Euler angles
# are fine here because each joint moves within a small range where
# gimbal lock cannot occur.

_REST_POSE = {
    'hips_joint':        (0, 0, 0),
    'spine_joint':       (0, 0, 0),
    'neck_joint':        (0, 0, 0),
    'head_joint':        (0, 0, 0),
    'shoulder_L_joint':  (0, 0, 0),
    'elbow_L_joint':     (0, 0, 0),
    'hand_L_joint':      (0, 0, 0),
    'shoulder_R_joint':  (0, 0, 0),
    'elbow_R_joint':     (0, 0, 0),
    'hand_R_joint':      (0, 0, 0),
    'hip_L_joint':       (0, 0, 0),
    'knee_L_joint':      (0, 0, 0),
    'foot_L_joint':      (0, 0, 0),
    'hip_R_joint':       (0, 0, 0),
    'knee_R_joint':      (0, 0, 0),
    'foot_R_joint':      (0, 0, 0),
}


class Player:
    """Third-person player with hierarchical skeleton and procedural animation."""

    MOVE_SPEED = 8.0
    HEIGHT_OFFSET = 0.0
    ACCEL_SMOOTH = 10.0
    DECEL_SMOOTH = 8.0
    TURN_SMOOTH = 10.0

    # Attack timing
    ATTACK_WINDUP = 0.12     # seconds — wind-up phase
    ATTACK_SLASH = 0.15      # seconds — slash phase
    ATTACK_RECOVERY = 0.20   # seconds — recovery phase
    ATTACK_DURATION = ATTACK_WINDUP + ATTACK_SLASH + ATTACK_RECOVERY

    # Blend speeds (higher = faster transition)
    BLEND_SPEED_NORMAL = 8.0
    BLEND_SPEED_ATTACK = 14.0

    # Sword trail
    TRAIL_MAX_POINTS = 12
    TRAIL_WIDTH = 0.15

    def __init__(self, planet, parent_np):
        self.planet = planet

        # Start at the "north pole"
        start_pos = Vec3(0, 0, planet.radius + self.HEIGHT_OFFSET)
        self.position = start_pos

        # Initial facing
        self._forward = Vec3(0, 1, 0)
        _, self._forward, _ = tangent_frame(
            self.position, self._forward, planet.center
        )

        # Smoothed velocity
        self._velocity = Vec3(0, 0, 0)

        # ── Build hierarchical skeleton ───────────────────────────────
        self.node = make_player_mesh()
        self.node.reparentTo(parent_np)

        # Cache references to all joint NodePaths for fast animation.
        # find() searches the scene-graph subtree by name.
        self._joints = {}
        for name in _REST_POSE:
            found = self.node.find('**/' + name)
            if not found.isEmpty():
                self._joints[name] = found

        # ── Sword — attached to right hand joint ─────────────────────
        # The blade mesh extends along local +Z.  The arm chain also
        # runs along Z, so we pitch the sword 90° to redirect the blade
        # along +Y (the player's forward axis).  This ensures the sharp
        # end leads during a forward thrust.
        self._sword_root = make_weapon_mesh()
        hand_R = self._joints.get('hand_R_joint')
        if hand_R:
            self._sword_root.reparentTo(hand_R)
            self._sword_root.setPos(0, 0, 0)
            self._sword_root.setHpr(0, -90, 0)  # pitch blade forward (+Y)
        else:
            self._sword_root.reparentTo(self.node)
            self._sword_root.setPos(0.5, 0.2, 1.1)

        # ── Animation state ──────────────────────────────────────────
        self._anim_state = AnimState.IDLE
        self._anim_time = 0.0
        self._state_time = 0.0  # time spent in current state

        # Current blended joint angles: {name: [h, p, r]}
        self._current_angles = {n: list(v) for n, v in _REST_POSE.items()}

        # ── Attack state ─────────────────────────────────────────────
        self._attacking = False
        self._attack_timer = 0.0

        # ── Hit react ────────────────────────────────────────────────
        self._hit_reacting = False
        self._hit_timer = 0.0
        self._hit_duration = 0.35

        # ── Sword trail ──────────────────────────────────────────────
        self._trail_points = []  # list of (pos_base, pos_tip, age)
        self._trail_node = self._make_trail_node(parent_np)

        self._update_transform()

    # ------------------------------------------------------------------
    # Trail mesh setup
    # ------------------------------------------------------------------

    def _make_trail_node(self, parent):
        """
        Pre-allocate a dynamic triangle-strip mesh for the sword trail.

        CG concept — Dynamic mesh updates:
          The trail geometry is rebuilt every frame from a ring buffer of
          recent blade positions.  Using Geom.UHDynamic hints the GPU
          driver to place the vertex buffer in write-friendly memory.
        """
        fmt = GeomVertexFormat.getV3c4()
        self._trail_vdata = GeomVertexData('trail', fmt, Geom.UHDynamic)
        self._trail_vdata.setNumRows(self.TRAIL_MAX_POINTS * 2)
        self._trail_geom = Geom(self._trail_vdata)
        self._trail_prim = GeomTristrips(Geom.UHDynamic)
        self._trail_geom.addPrimitive(self._trail_prim)
        gn = GeomNode('sword_trail')
        gn.addGeom(self._trail_geom)
        np = parent.attachNewNode(gn)
        np.setTransparency(TransparencyAttrib.MAlpha)
        np.setBin('transparent', 15)
        np.setDepthWrite(False)
        np.setLightOff()
        np.setTwoSided(True)
        return np

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt, input_vec, camera_forward_hint):
        """
        Move the player on the sphere surface and advance animations.

        *input_vec*: Vec3 with x = right/left, y = forward/back (WASD).
        *camera_forward_hint*: camera's look direction on the tangent plane.
        """
        up = self.planet.local_up(self.position)

        # Build tangent frame from camera hint
        right, fwd, up = tangent_frame(
            self.position, camera_forward_hint, self.planet.center
        )

        # Desired movement
        has_input = input_vec.length() > 0.01
        if has_input:
            target_dir = fwd * input_vec.y + right * input_vec.x
            if target_dir.length() > 0.01:
                target_dir = normalized(target_dir)
            target_vel = target_dir * self.MOVE_SPEED
        else:
            target_vel = Vec3(0, 0, 0)

        # Smooth velocity
        smooth = self.ACCEL_SMOOTH if has_input else self.DECEL_SMOOTH
        t = min(1.0, smooth * dt)
        self._velocity = lerp_vec3(self._velocity, target_vel, t)

        # Apply velocity
        if self._velocity.length() > 0.05:
            self.position += self._velocity * dt
            self.position = snap_to_surface(
                self.position, self.planet.radius, self.HEIGHT_OFFSET,
                self.planet.center
            )
            new_up = self.planet.local_up(self.position)
            self._velocity = self._velocity - new_up * self._velocity.dot(new_up)

            move_dir = normalized(self._velocity)
            _, new_fwd, _ = tangent_frame(
                self.position, move_dir, self.planet.center
            )
            turn_t = min(1.0, self.TURN_SMOOTH * dt)
            self._forward = slerp_vec3(self._forward, new_fwd, turn_t)
            _, self._forward, _ = tangent_frame(
                self.position, self._forward, self.planet.center
            )

        # Advance animation timers
        self._anim_time += dt
        self._state_time += dt

        # Determine animation state
        self._update_anim_state(has_input, dt)

        # Compute target pose for current state
        target = self._compute_target_pose(has_input)

        # Blend current angles toward target
        blend_speed = self.BLEND_SPEED_ATTACK if self._attacking else self.BLEND_SPEED_NORMAL
        bt = min(1.0, blend_speed * dt)
        for name in self._current_angles:
            cur = self._current_angles[name]
            tgt = target.get(name, (0, 0, 0))
            cur[0] = lerp_float(cur[0], tgt[0], bt)
            cur[1] = lerp_float(cur[1], tgt[1], bt)
            cur[2] = lerp_float(cur[2], tgt[2], bt)

        # Apply blended angles to joints
        for name, joint in self._joints.items():
            h, p, r = self._current_angles[name]
            joint.setHpr(h, p, r)

        # Update sword trail
        self._update_trail(dt)

        # Update world transform
        self._update_transform()

    # ------------------------------------------------------------------
    # Animation state machine
    # ------------------------------------------------------------------

    def _update_anim_state(self, has_input, dt):
        """Transition between animation states based on gameplay."""
        if self._hit_reacting:
            self._hit_timer += dt
            if self._hit_timer >= self._hit_duration:
                self._hit_reacting = False
                self._set_state(AnimState.IDLE)
            else:
                self._set_state(AnimState.HIT_REACT)
            return

        if self._attacking:
            self._attack_timer += dt
            if self._attack_timer >= self.ATTACK_DURATION:
                self._attacking = False
                self._set_state(AnimState.IDLE if not has_input else AnimState.WALKING)
            else:
                self._set_state(AnimState.ATTACKING)
            return

        if has_input:
            self._set_state(AnimState.WALKING)
        else:
            self._set_state(AnimState.IDLE)

    def _set_state(self, new_state):
        if new_state != self._anim_state:
            self._anim_state = new_state
            self._state_time = 0.0

    # ------------------------------------------------------------------
    # Pose computation — procedural joint angles per state
    # ------------------------------------------------------------------

    def _compute_target_pose(self, has_input):
        """
        Compute the target HPR for every joint based on the current
        animation state and elapsed time.

        CG concept — Procedural animation:
          Instead of sampling pre-recorded keyframes, we compute joint
          angles analytically using trigonometric functions of time.
          sin/cos produce smooth, periodic oscillations perfect for
          walk cycles and idle breathing.  Easing functions reshape
          the timing curves for attack wind-ups and recoveries.
        """
        state = self._anim_state
        t = self._anim_time  # global time (for continuous oscillation)
        st = self._state_time  # time in current state

        pose = {n: list(v) for n, v in _REST_POSE.items()}

        if state == AnimState.IDLE:
            self._pose_idle(pose, t)
        elif state == AnimState.WALKING:
            speed = self._velocity.length() / self.MOVE_SPEED
            self._pose_walk(pose, t, speed)
        elif state == AnimState.ATTACKING:
            self._pose_attack(pose, st)
        elif state == AnimState.HIT_REACT:
            self._pose_hit_react(pose, st)

        return pose

    def _pose_idle(self, pose, t):
        """
        Idle pose: sword held forward in a ready stance.

        The sword arm (right) is held out in front so the blade is
        visible ahead of the character, matching the combat hitbox
        direction.  Subtle breathing and weight shift add life.
        """
        # Breathing: spine pitches forward/back ~2° at 0.8 Hz
        breath = math.sin(t * 0.8 * 2 * math.pi) * 2.0
        pose['spine_joint'] = (0, breath, 0)

        # Weight shift: hips sway side-to-side ~1.5° at 0.3 Hz
        sway = math.sin(t * 0.3 * 2 * math.pi) * 1.5
        pose['hips_joint'] = (0, 0, sway)

        # Head subtle look-around
        head_h = math.sin(t * 0.2 * 2 * math.pi) * 3.0
        pose['head_joint'] = (head_h, 0, 0)

        # Left arm relaxed at side
        pose['shoulder_L_joint'] = (0, 5, 0)
        pose['elbow_L_joint'] = (0, -8, 0)

        # Right arm (sword arm) held forward in a guard stance
        # Shoulder pitched forward so sword extends ahead of the player
        pose['shoulder_R_joint'] = (0, 50, 0)
        pose['elbow_R_joint'] = (0, -30, 0)

        # Subtle leg stance
        pose['hip_L_joint'] = (0, 2, 0)
        pose['hip_R_joint'] = (0, 2, 0)
        pose['knee_L_joint'] = (0, -3, 0)
        pose['knee_R_joint'] = (0, -3, 0)

    def _pose_walk(self, pose, t, speed):
        """
        Walk cycle: hip-driven pendulum motion with counter-swing arms.

        CG concept — Procedural walk cycle:
          The walk cycle is a phase-locked system of sinusoidal
          oscillations.  The left leg and right arm are 180° out of
          phase with the right leg and left arm, producing the natural
          contra-lateral pattern of human locomotion.

          walk_phase = time × frequency × 2π
          leg_swing  = sin(phase) × amplitude
          arm_swing  = sin(phase + π) × amplitude   ← counter-swing
          torso_bob  = |sin(phase)| × bounce_height  ← double-frequency
        """
        freq = 2.5  # steps per second
        amp = min(1.0, speed) if speed > 0.1 else 0.0
        phase = t * freq * 2 * math.pi

        # Leg swing amplitude (degrees)
        leg_swing = 25.0 * amp
        knee_bend = 20.0 * amp

        # Left leg (sin phase)
        l_hip = math.sin(phase) * leg_swing
        l_knee = -abs(math.sin(phase)) * knee_bend - 5  # always slightly bent
        pose['hip_L_joint'] = (0, l_hip, 0)
        pose['knee_L_joint'] = (0, l_knee, 0)

        # Right leg (opposite phase)
        r_hip = math.sin(phase + math.pi) * leg_swing
        r_knee = -abs(math.sin(phase + math.pi)) * knee_bend - 5
        pose['hip_R_joint'] = (0, r_hip, 0)
        pose['knee_R_joint'] = (0, r_knee, 0)

        # Left arm counter-swing (free hand)
        arm_swing = 20.0 * amp
        pose['shoulder_L_joint'] = (0, -math.sin(phase) * arm_swing, 0)
        pose['elbow_L_joint'] = (0, -15 - abs(math.sin(phase)) * 10 * amp, 0)

        # Right arm (sword arm) stays forward in guard position with subtle bob
        sword_bob = math.sin(phase) * 5.0 * amp
        pose['shoulder_R_joint'] = (0, 50 + sword_bob, 0)
        pose['elbow_R_joint'] = (0, -30 + sword_bob * 0.5, 0)

        # Torso twist (subtle counter-rotation to legs)
        twist = math.sin(phase) * 4.0 * amp
        pose['spine_joint'] = (twist, 2.0, 0)

        # Hip bob (double frequency — bounces twice per step cycle)
        bob = abs(math.sin(phase)) * 2.0 * amp
        pose['hips_joint'] = (0, bob, 0)

        # Head stays mostly level (counter-rotates against spine twist)
        pose['head_joint'] = (-twist * 0.5, 0, 0)

    def _pose_attack(self, pose, st):
        """
        3-phase forward thrust attack: wind-up → thrust → recovery.

        The sword arm pulls back then thrusts *forward* (along +Y / the
        player's facing direction) so the blade tip reaches the hitbox
        position (forward * 3 + up * 1 from the player).

        CG concept — Easing functions in animation:
          Raw linear interpolation produces robotic motion.  By remapping
          the progress parameter through polynomial easing functions, we
          get natural acceleration/deceleration:
            - Wind-up uses ease_in_quad (slow start → builds tension)
            - Thrust uses ease_out_cubic (explosive start → decelerates)
            - Recovery uses ease_in_out_quad (smooth settle to rest)
        """
        wu = self.ATTACK_WINDUP
        sl = self.ATTACK_SLASH
        rc = self.ATTACK_RECOVERY

        if st < wu:
            # Phase 1: Wind-up — pull sword arm back and up
            p = ease_in_quad(st / wu)
            # Torso leans back slightly to coil
            pose['spine_joint'] = (0, -8 * p, 0)
            # Right arm raises up and back (pitch = shoulder rotation forward/back)
            # Positive pitch = forward, negative = back
            pose['shoulder_R_joint'] = (0, -50 * p, 0)
            pose['elbow_R_joint'] = (0, -70 * p, 0)
            # Slight crouch for power
            pose['hips_joint'] = (0, 5 * p, 0)
            pose['knee_L_joint'] = (0, -8 * p, 0)
            pose['knee_R_joint'] = (0, -8 * p, 0)
            # Left arm braces forward
            pose['shoulder_L_joint'] = (0, 15 * p, 0)
            pose['elbow_L_joint'] = (0, -20 * p, 0)

        elif st < wu + sl:
            # Phase 2: Thrust — explosive forward swing toward hitbox
            p = ease_out_cubic((st - wu) / sl)
            # Torso lunges forward
            pose['spine_joint'] = (0, -8 + 20 * p, 0)
            # Right arm swings forward and extends (pitch goes positive = forward)
            pose['shoulder_R_joint'] = (0, -50 + 120 * p, 0)
            pose['elbow_R_joint'] = (0, -70 + 55 * p, 0)
            # Legs push off
            pose['hips_joint'] = (0, 5 - 3 * p, 0)
            pose['knee_L_joint'] = (0, -8 + 4 * p, 0)
            pose['knee_R_joint'] = (0, -8 + 4 * p, 0)
            # Step forward with right leg
            pose['hip_R_joint'] = (0, 15 * p, 0)
            # Left arm counterbalances back
            pose['shoulder_L_joint'] = (0, 15 - 25 * p, 0)
            pose['elbow_L_joint'] = (0, -20 + 10 * p, 0)

        else:
            # Phase 3: Recovery — ease back to rest
            p = ease_in_out_quad((st - wu - sl) / rc)
            r = 1 - p  # remaining factor
            pose['spine_joint'] = (0, 12 * r, 0)
            pose['shoulder_R_joint'] = (0, 70 * r, 0)
            pose['elbow_R_joint'] = (0, -15 * r, 0)
            pose['hips_joint'] = (0, 2 * r, 0)
            pose['hip_R_joint'] = (0, 15 * r, 0)
            pose['shoulder_L_joint'] = (0, -10 * r, 0)
            pose['knee_L_joint'] = (0, -4 * r, 0)
            pose['knee_R_joint'] = (0, -4 * r, 0)

    def _pose_hit_react(self, pose, st):
        """
        Hit reaction: flinch backward then recover.
        Uses elastic ease-out for a snappy, impactful feel.
        """
        p = min(1.0, st / self._hit_duration)
        # Flinch peaks at p~0.3, then settles
        if p < 0.3:
            flinch = ease_in_quad(p / 0.3)
        else:
            flinch = 1.0 - ease_out_elastic((p - 0.3) / 0.7)
            flinch = max(0, flinch)

        pose['spine_joint'] = (0, -15 * flinch, 5 * flinch)
        pose['hips_joint'] = (0, -5 * flinch, 0)
        pose['head_joint'] = (0, -10 * flinch, 0)
        pose['shoulder_L_joint'] = (10 * flinch, 15 * flinch, 0)
        pose['shoulder_R_joint'] = (-10 * flinch, 15 * flinch, 0)

    # ------------------------------------------------------------------
    # Sword swing interface (called by CombatSystem)
    # ------------------------------------------------------------------

    def start_swing(self):
        """Trigger the attack animation."""
        if self._attacking:
            return
        self._attacking = True
        self._attack_timer = 0.0
        self._set_state(AnimState.ATTACKING)
        self._trail_points.clear()

    @property
    def is_swinging(self):
        return self._attacking

    def take_hit(self):
        """Trigger the hit-react animation."""
        if self._attacking:
            return
        self._hit_reacting = True
        self._hit_timer = 0.0
        self._set_state(AnimState.HIT_REACT)

    # ------------------------------------------------------------------
    # Sword trail
    # ------------------------------------------------------------------

    def _get_blade_tip_world(self):
        """Get the blade tip position in world space."""
        # The blade mesh extends along +Z, but the sword NodePath is
        # pitched -90° so +Z maps to +Y in the hand's local frame.
        # We query in the sword's own coordinate space (pre-rotation).
        tip_local = Vec3(0, 0, 1.42)
        base_local = Vec3(0, 0, 0.2)
        try:
            parent = self.node.getParent()
            tip_world = parent.getRelativePoint(self._sword_root, tip_local)
            base_world = parent.getRelativePoint(self._sword_root, base_local)
            return base_world, tip_world
        except Exception:
            return None, None

    def _update_trail(self, dt):
        """
        Record blade position and rebuild the trail quad-strip mesh.

        CG concept — Dynamic quad-strip:
          Each frame we push two new vertices (blade base + tip) and
          connect them to the previous pair as a triangle strip:
              v0─v2─v4─...
              │╲ │╲ │╲
              v1─v3─v5─...
          The strip is rebuilt from scratch each frame using Geom.UHDynamic,
          which tells the GPU to expect frequent updates.
        """
        # Age existing points
        new_pts = []
        for base, tip, age in self._trail_points:
            age += dt
            if age < 0.2:  # trail lifetime
                new_pts.append((base, tip, age))
        self._trail_points = new_pts

        # Add new point if attacking (during slash phase)
        if self._attacking:
            at = self._attack_timer
            wu = self.ATTACK_WINDUP
            if at >= wu * 0.5:  # start trail slightly before slash
                base, tip = self._get_blade_tip_world()
                if base is not None:
                    self._trail_points.append((base, tip, 0.0))

        # Cap length
        if len(self._trail_points) > self.TRAIL_MAX_POINTS:
            self._trail_points = self._trail_points[-self.TRAIL_MAX_POINTS:]

        # Rebuild mesh
        self._rebuild_trail_mesh()

    def _rebuild_trail_mesh(self):
        """Rebuild the trail triangle strip geometry from recorded points."""
        pts = self._trail_points
        n = len(pts)

        self._trail_geom.clearPrimitives()

        if n < 2:
            self._trail_node.hide()
            return

        self._trail_node.show()
        self._trail_vdata.setNumRows(n * 2)
        wv = GeomVertexWriter(self._trail_vdata, 'vertex')
        wc = GeomVertexWriter(self._trail_vdata, 'color')

        prim = GeomTristrips(Geom.UHDynamic)

        for i, (base, tip, age) in enumerate(pts):
            alpha = max(0, 1.0 - age / 0.2) * 0.6
            color = Vec4(0.5, 0.7, 1.0, alpha)  # blue-white trail
            wv.addData3(base)
            wc.addData4(color)
            wv.addData3(tip)
            wc.addData4(color)

        for i in range(n * 2):
            prim.addVertex(i)
        prim.closePrimitive()

        self._trail_geom.addPrimitive(prim)

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _update_transform(self):
        """Apply position and quaternion orientation to the root scene node."""
        self.node.setPos(self.position)
        up = self.planet.local_up(self.position)
        quat = quat_from_forward_up(self._forward, up)
        self.node.setQuat(quat)

    def get_forward(self):
        """Current forward direction on the tangent plane."""
        return Vec3(self._forward)

    def get_up(self):
        """Current local up (surface normal)."""
        return self.planet.local_up(self.position)

    def get_right(self):
        """Current local right vector."""
        return normalized(self.get_up().cross(self._forward))
