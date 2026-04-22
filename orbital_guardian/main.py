#!/usr/bin/env python3
"""
Orbital Guardian — main entry point.

A micro-planet defender game built with Panda3D for a Computer Graphics
university course.  Demonstrates:
  - Procedural mesh generation (icosphere, low-poly humanoid, starfield)
  - Custom GLSL shaders (Blinn-Phong, fresnel atmosphere, pulsing emissive)
  - Spherical gravity and tangent-plane player movement
  - Quaternion-based orientation (no gimbal lock)
  - Third-person camera with spherical orbit
  - Real-time billboard particle effects
  - Game state machine (menu → play → game over)

Run:  python main.py
"""

import sys
import os
import math

# Ensure the project root is on the Python path so that subpackage imports
# (e.g. ``from core.planet import Planet``) work regardless of cwd.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    Vec3, Vec4, Shader, WindowProperties, ClockObject,
    AntialiasAttrib, TransparencyAttrib, TextNode, loadPrcFileData,
    KeyboardButton, MouseButton,
)

# Panda3D config — must be set before ShowBase.__init__
loadPrcFileData('', 'window-title Orbital Guardian')
loadPrcFileData('', 'framebuffer-multisample 1')
loadPrcFileData('', 'multisamples 4')

from core.planet import Planet
from core.player import Player
from core.camera_rig import CameraRig
from core.meteor_spawner import MeteorSpawner
from core.combat import CombatSystem
from core.impact_predictor import ImpactPredictor
from graphics.procedural_meshes import make_skybox_mesh
from graphics.particles import ParticleManager
from graphics.lighting import setup_lighting
from ui.hud import HUD


# ======================================================================
# Game states
# ======================================================================
STATE_MENU = 'menu'
STATE_PLAYING = 'playing'
STATE_GAME_OVER = 'game_over'


class OrbitalGuardian(ShowBase):
    """Main application class."""

    def __init__(self):
        super().__init__()

        # Basic window setup
        self.setBackgroundColor(0.02, 0.02, 0.08, 1)
        self.render.setAntialias(AntialiasAttrib.MAuto)

        # Disable default mouse camera control
        self.disableMouse()

        # Global clock
        self.clock = ClockObject.getGlobalClock()
        self.clock.setMode(ClockObject.MLimited)
        self.clock.setFrameRate(60)

        # Elapsed time for shaders
        self._time = 0.0

        # Input state
        self._keys = {
            'w': False, 's': False, 'a': False, 'd': False,
            'attack': False,
        }
        self._setup_input()

        # State machine
        self.state = STATE_MENU
        self._menu_text = None
        self._gameover_text = None

        # Core systems (created on game start)
        self._planet = None
        self._player = None
        self._camera_rig = None
        self._spawner = None
        self._combat = None
        self._particles = None
        self._impact_predictor = None
        self._hud = None
        self._lights = None
        self._skybox = None

        # Shader cache
        self._planet_shader = None
        self._atmos_shader = None
        self._meteor_shader = None

        # Trail spawn timer
        self._trail_timer = 0.0

        # Show menu
        self._enter_menu()

        # Main update task
        self.taskMgr.add(self._update, 'game_update')

    # ------------------------------------------------------------------
    # Input — uses multiple backends to work around Wayland/XWayland issues
    # ------------------------------------------------------------------

    def _setup_input(self):
        """
        Panda3D's X11 input often fails on Wayland compositors.
        We set up three layers:
          1. Panda3D event system (works on X11)
          2. Panda3D polling via mouseWatcherNode (backup)
          3. Raw /dev/input evdev reading (Wayland fallback)
        """
        self._input_debug_timer = 0.0
        self._attack_pressed = False

        # --- Layer 1: Panda3D events (fallback — unreliable on Wayland) ---
        # We do NOT bind WASD here because key-up events don't fire on
        # Wayland, causing keys to get stuck.  Movement is handled purely
        # through evdev/polling held-state in _poll_input().
        self.accept('space', self._on_action_press)
        self.accept('enter', self._on_action_press)
        self.accept('return', self._on_action_press)
        self.accept('mouse1', self._on_action_press)
        self.accept('r', self._on_r_press)
        self.accept('escape', sys.exit)

        # --- Layer 2: polling button handles ---
        self._btn_map = {
            'w': KeyboardButton.asciiKey('w'),
            's': KeyboardButton.asciiKey('s'),
            'a': KeyboardButton.asciiKey('a'),
            'd': KeyboardButton.asciiKey('d'),
            'space': KeyboardButton.space(),
            'enter': KeyboardButton.enter(),
            'r': KeyboardButton.asciiKey('r'),
            'escape': KeyboardButton.escape(),
        }
        self._mouse_btn = MouseButton.one()
        self._prev_poll = {k: False for k in self._btn_map}
        self._prev_poll['mouse1'] = False

        # --- Layer 3: evdev keyboard (Linux raw input, works on Wayland) ---
        self._evdev_thread = None
        self._evdev_keys = {}  # key_name -> pressed (bool)
        self._evdev_prev = {}
        self._start_evdev_reader()

        # Diagnostic: print input system status once
        print('[INPUT] Setting up input layers...')
        print(f'[INPUT]   mouseWatcherNode: {self.mouseWatcherNode}')
        print(f'[INPUT]   evdev thread: {"running" if self._evdev_thread and self._evdev_thread.is_alive() else "not available"}')
        print(f'[INPUT]   Panda3D events: registered')

    def _start_evdev_reader(self):
        """
        Start a background thread that reads keyboard events from
        /dev/input/event* using raw evdev protocol.  This works on
        Wayland where X11 keyboard events are not delivered.
        """
        import threading
        import struct
        import glob

        def find_keyboard():
            """Find the keyboard device by checking capabilities."""
            for dev_path in sorted(glob.glob('/dev/input/event*')):
                try:
                    with open(dev_path, 'rb') as f:
                        # Read device name
                        import fcntl
                        import array
                        buf = array.array('B', [0] * 256)
                        try:
                            fcntl.ioctl(f, 0x80FF4506, buf)  # EVIOCGNAME
                            name = buf.tobytes().split(b'\x00')[0].decode('utf-8', errors='ignore')
                        except Exception:
                            name = ''
                        # Check if device has KEY events (EV_KEY = 1)
                        # EVIOCGBIT(0, size) = 0x80084520 for 8 bytes
                        ev_bits = array.array('B', [0] * 8)
                        try:
                            fcntl.ioctl(f, 0x80084520, ev_bits)
                        except Exception:
                            continue
                        # Bit 1 (EV_KEY) set?
                        if ev_bits[0] & 0x02:
                            # Check it has typical keyboard keys (KEY_SPACE=57)
                            # EVIOCGBIT(EV_KEY, size) = 0x80604521
                            key_bits = array.array('B', [0] * 96)
                            try:
                                fcntl.ioctl(f, 0x80604521, key_bits)
                            except Exception:
                                continue
                            # KEY_SPACE=57 → byte 7, bit 1
                            if key_bits[57 // 8] & (1 << (57 % 8)):
                                return dev_path
                except (PermissionError, OSError):
                    continue
            return None

        # evdev key code → our key name mapping
        # See linux/input-event-codes.h
        evdev_keymap = {
            17: 'w', 31: 's', 30: 'a', 32: 'd',
            57: 'space', 28: 'enter',  # KEY_ENTER
            19: 'r', 1: 'escape',
            96: 'enter',  # KEY_KPENTER (numpad)
        }

        def reader_loop(dev_path):
            """Read evdev events in a loop. Each event is 24 bytes on 64-bit."""
            # struct input_event: time (16 bytes), type (2), code (2), value (4)
            EVENT_SIZE = 24
            EVENT_FMT = 'llHHi'
            try:
                with open(dev_path, 'rb') as f:
                    while True:
                        data = f.read(EVENT_SIZE)
                        if len(data) < EVENT_SIZE:
                            break
                        _, _, ev_type, code, value = struct.unpack(EVENT_FMT, data)
                        if ev_type == 1:  # EV_KEY
                            key_name = evdev_keymap.get(code)
                            if key_name:
                                # value: 0=release, 1=press, 2=repeat
                                self._evdev_keys[key_name] = (value > 0)
            except Exception as e:
                print(f'[INPUT] evdev reader stopped: {e}')

        kb_device = find_keyboard()
        if kb_device:
            print(f'[INPUT]   evdev keyboard found: {kb_device}')
            self._evdev_thread = threading.Thread(
                target=reader_loop, args=(kb_device,), daemon=True
            )
            self._evdev_thread.start()
        else:
            print('[INPUT]   evdev: no keyboard device found (may need permissions)')
            print('[INPUT]   Try: sudo usermod -aG input $USER  (then re-login)')

    def _set_key(self, key, value):
        self._keys[key] = value

    def _on_action_press(self):
        """Handles space/enter/click from Panda3D event system."""
        if self.state == STATE_MENU:
            try:
                self._start_game()
            except Exception:
                import traceback
                traceback.print_exc()
        elif self.state == STATE_PLAYING:
            self._keys['attack'] = True

    def _on_r_press(self):
        if self.state == STATE_GAME_OVER:
            try:
                self._cleanup_game()
                self._start_game()
            except Exception:
                import traceback
                traceback.print_exc()

    def _is_down_poll(self, name):
        """Check button via mouseWatcherNode polling."""
        if not self.mouseWatcherNode:
            return False
        btn = self._btn_map.get(name)
        if btn:
            return self.mouseWatcherNode.isButtonDown(btn)
        if name == 'mouse1':
            return self.mouseWatcherNode.isButtonDown(self._mouse_btn)
        return False

    def _is_down_evdev(self, name):
        """Check button via raw evdev state."""
        return self._evdev_keys.get(name, False)

    def _is_down(self, name):
        """Combined input: evdev is primary, polling is backup.
        We do NOT include self._keys (Panda3D events) for held-state checks
        because on Wayland the key-up events never fire, causing stuck keys."""
        return self._is_down_evdev(name) or self._is_down_poll(name)

    def _just_pressed_evdev(self, name):
        """Edge detection on evdev state."""
        current = self._is_down_evdev(name)
        prev = self._evdev_prev.get(name, False)
        return current and not prev

    def _just_pressed_poll(self, name):
        """Edge detection on poll state."""
        current = self._is_down_poll(name)
        prev = self._prev_poll.get(name, False)
        return current and not prev

    def _poll_input(self):
        """
        Called every frame.  Checks all input layers for key state and
        triggers game actions.
        """
        # Movement keys — read directly from evdev/polling held state.
        # These reflect real-time press/release with no sticking.
        self._keys['w'] = self._is_down('w')
        self._keys['s'] = self._is_down('s')
        self._keys['a'] = self._is_down('a')
        self._keys['d'] = self._is_down('d')
        self._keys['attack'] = False  # reset each frame; set below if pressed

        # One-shot actions via polling/evdev edge detection
        space_press = self._just_pressed_poll('space') or self._just_pressed_evdev('space')
        enter_press = self._just_pressed_poll('enter') or self._just_pressed_evdev('enter')
        r_press = self._just_pressed_poll('r') or self._just_pressed_evdev('r')
        esc_press = self._just_pressed_poll('escape') or self._just_pressed_evdev('escape')
        mouse_press = self._just_pressed_poll('mouse1')

        if space_press or enter_press:
            if self.state == STATE_MENU:
                try:
                    self._start_game()
                except Exception:
                    import traceback
                    traceback.print_exc()
            elif self.state == STATE_PLAYING:
                self._keys['attack'] = True

        if mouse_press and self.state == STATE_PLAYING:
            self._keys['attack'] = True

        if r_press and self.state == STATE_GAME_OVER:
            try:
                self._cleanup_game()
                self._start_game()
            except Exception:
                import traceback
                traceback.print_exc()

        if esc_press:
            sys.exit()

        # Save states for edge detection next frame
        for name in self._btn_map:
            self._prev_poll[name] = self._is_down_poll(name)
        self._prev_poll['mouse1'] = self._is_down_poll('mouse1')
        for name in ('w', 's', 'a', 'd', 'space', 'enter', 'r', 'escape'):
            self._evdev_prev[name] = self._is_down_evdev(name)

        # Periodic debug print (once every 3 seconds)
        self._input_debug_timer += self.clock.getDt()
        if self._input_debug_timer > 3.0:
            self._input_debug_timer = 0.0
            evdev_any = any(self._evdev_keys.values())
            poll_test = self._is_down_poll('space')
            print(f'[INPUT] state={self.state} | evdev_active={bool(self._evdev_thread and self._evdev_thread.is_alive())} | evdev_keys_pressed={evdev_any} | poll_space={poll_test}')

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _enter_menu(self):
        self.state = STATE_MENU
        self._menu_text = OnscreenText(
            text='ORBITAL GUARDIAN\n\nDefend your planet from meteors!\n\n'
                 'WASD to move   |   SPACE / Click to attack\n\n'
                 'Press SPACE or ENTER to Start',
            pos=(0, 0.1), scale=0.07,
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.7),
            parent=self.aspect2d, mayChange=False
        )

    def _start_game(self):
        if self._menu_text:
            self._menu_text.destroy()
            self._menu_text = None
        if self._gameover_text:
            self._gameover_text.destroy()
            self._gameover_text = None

        self.state = STATE_PLAYING

        # --- Create world ---
        self._planet = Planet(self.render)
        self._player = Player(self._planet, self.render)
        self._camera_rig = CameraRig(self.camera, self._player, self._planet)
        self._spawner = MeteorSpawner(self._planet, self.render)
        self._combat = CombatSystem(self._player, self._spawner)
        self._particles = ParticleManager(self.render)
        self._impact_predictor = ImpactPredictor(self._planet, self.render)

        # Skybox
        self._skybox = make_skybox_mesh(radius=500.0, subdivisions=2)
        self._skybox.reparentTo(self.render)

        # Lighting
        self._lights = setup_lighting(self.render)

        # Load and apply shaders
        self._load_shaders()

        # HUD
        self._hud = HUD(self.aspect2d)

        self._time = 0.0

    def _enter_game_over(self):
        self.state = STATE_GAME_OVER
        score = self._combat.score if self._combat else 0
        wave = self._spawner.wave if self._spawner else 0
        self._hud.hide()
        self._gameover_text = OnscreenText(
            text=f'GAME OVER\n\nYour planet was destroyed!\n\n'
                 f'Score: {score}   |   Wave: {wave}\n\n'
                 f'Press R to Restart',
            pos=(0, 0.1), scale=0.07,
            fg=(1, 0.3, 0.3, 1), shadow=(0, 0, 0, 0.7),
            parent=self.aspect2d, mayChange=False
        )

    def _cleanup_game(self):
        """Tear down all game objects for a clean restart."""
        if self._spawner:
            self._spawner.destroy_all()
        if self._particles:
            self._particles.cleanup()
        if self._impact_predictor:
            self._impact_predictor.cleanup()
        if self._hud:
            self._hud.cleanup()
        if self._gameover_text:
            self._gameover_text.destroy()
            self._gameover_text = None
        # Remove scene nodes
        if self._planet:
            self._planet.node.removeNode()
            self._planet.atmosphere.removeNode()
        if self._player:
            self._player.node.removeNode()
            if hasattr(self._player, '_trail_node'):
                self._player._trail_node.removeNode()
        if self._skybox:
            self._skybox.removeNode()
        # Remove lights
        if self._lights:
            for lnp in self._lights.values():
                self.render.clearLight(lnp)
                lnp.removeNode()
        self._planet = None
        self._player = None
        self._camera_rig = None
        self._spawner = None
        self._combat = None
        self._particles = None
        self._impact_predictor = None
        self._hud = None
        self._lights = None
        self._skybox = None

    # ------------------------------------------------------------------
    # Shaders
    # ------------------------------------------------------------------

    def _load_shaders(self):
        """Load GLSL shader programs and attach them to scene objects."""
        shader_dir = os.path.join(_PROJECT_ROOT, 'graphics', 'shaders')

        # Planet shader
        self._planet_shader = Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(shader_dir, 'planet.vert.glsl'),
            fragment=os.path.join(shader_dir, 'planet.frag.glsl'),
        )
        self._planet.node.setShader(self._planet_shader)
        self._planet.node.setShaderInput('lightDir', Vec3(0.58, 0.58, -0.58))
        self._planet.node.setShaderInput('lightColor', Vec3(1.0, 0.95, 0.8))
        self._planet.node.setShaderInput('ambientColor', Vec3(0.15, 0.15, 0.25))

        # Atmosphere shader
        self._atmos_shader = Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(shader_dir, 'atmosphere.vert.glsl'),
            fragment=os.path.join(shader_dir, 'atmosphere.frag.glsl'),
        )
        self._planet.atmosphere.setShader(self._atmos_shader)

        # Meteor shader (applied per-meteor in _apply_meteor_shader)
        self._meteor_shader = Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(shader_dir, 'meteor.vert.glsl'),
            fragment=os.path.join(shader_dir, 'meteor.frag.glsl'),
        )

    def _apply_meteor_shader(self, meteor_node):
        """Apply the meteor shader to a newly spawned meteor."""
        meteor_node.setShader(self._meteor_shader)
        meteor_node.setShaderInput('lightDir', Vec3(0.58, 0.58, -0.58))
        meteor_node.setShaderInput('lightColor', Vec3(1.0, 0.95, 0.8))
        meteor_node.setShaderInput('ambientColor', Vec3(0.15, 0.15, 0.25))
        meteor_node.setShaderInput('time', self._time)

    def _update_shader_uniforms(self):
        """Push per-frame uniform updates to shaders."""
        cam_pos = self.camera.getPos(self.render)

        if self._planet:
            self._planet.node.setShaderInput('cameraPos', cam_pos)
            self._planet.atmosphere.setShaderInput('cameraPos', cam_pos)

        # Update meteor shader time + apply shader to new meteors
        for m in self._spawner.get_active_meteors():
            if m.node:
                if not m.node.getShader():
                    self._apply_meteor_shader(m.node)
                m.node.setShaderInput('time', self._time)

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def _update(self, task):
        dt = self.clock.getDt()
        self._time += dt

        # Poll raw input every frame (event system broken on Wayland)
        self._poll_input()

        if self.state == STATE_PLAYING:
            self._update_gameplay(dt)

        return task.cont

    def _update_gameplay(self, dt):
        # --- Player input ---
        input_vec = Vec3(0, 0, 0)
        if self._keys['w']:
            input_vec.y += 1
        if self._keys['s']:
            input_vec.y -= 1
        if self._keys['a']:
            input_vec.x += 1
        if self._keys['d']:
            input_vec.x -= 1
        if input_vec.length() > 1:
            input_vec = input_vec / input_vec.length()

        # Camera forward for player-relative movement
        cam_fwd = self._camera_rig.get_forward()

        # Player movement
        self._player.update(dt, input_vec, cam_fwd)

        # Camera follow
        self._camera_rig.update(dt)

        # Attack
        if self._keys['attack']:
            started = self._combat.try_attack()
            if started:
                # Swoosh particles
                self._particles.spawn_attack_swoosh(
                    self._player.position + self._player.get_forward() * 1.5
                    + self._player.get_up() * 1.0,
                    self._player.get_forward()
                )
            self._keys['attack'] = False  # one-shot per press

        # Combat update
        self._combat.update(dt)

        # Meteor spawner — returns (explosions, newly_embedded)
        explosions, newly_embedded = self._spawner.update(dt)

        # Fuse explosions damage the planet
        for hit_pos in explosions:
            destroyed = self._planet.take_damage(15)
            self._particles.spawn_impact(hit_pos, count=25)
            if destroyed:
                self._enter_game_over()
                return

        # Impact particles for newly embedded meteors (subtle thud)
        for pos in newly_embedded:
            self._particles.spawn_impact(pos, count=8)

        # Particles for combat kills
        for pos in self._combat.get_safe_destroys():
            # Green/blue burst for punching embedded meteors
            self._particles.spawn_explosion(pos)
        for pos in self._combat.get_falling_destroys():
            # Orange burst for hitting falling meteors (bonus)
            self._particles.spawn_explosion(pos)

        # Meteor trail particles (only for falling meteors, throttled)
        self._trail_timer -= dt
        if self._trail_timer <= 0:
            self._trail_timer = 0.05
            for m in self._spawner.get_falling_meteors():
                if m.node:
                    self._particles.spawn_trail(m.node.getPos(), m.velocity)

        # Particles
        self._particles.update(dt)

        # Impact prediction markers
        self._impact_predictor.update(
            self._spawner.get_active_meteors(), self._time
        )

        # Shader uniforms
        self._update_shader_uniforms()

        # HUD — basic stats
        health_frac = self._planet.health / self._planet.max_health
        self._hud.update(
            health_frac,
            self._combat.score,
            self._spawner.wave,
            self._combat.get_cooldown_fraction()
        )

        # HUD — directional arrows for off-screen meteors
        self._update_hud_arrows()

    def _update_hud_arrows(self):
        """
        Project each active meteor (or its impact point) to screen space.
        If off-screen, pass data to HUD for directional arrow rendering.
        """
        from core.meteor_spawner import MeteorState

        if not self.camLens or not self.cam:
            return

        meteor_data = []
        proj_mat = self.cam.getNetTransform().getInverse().getMat()
        lens_mat = self.camLens.getProjectionMat()

        for meteor in self._spawner.get_active_meteors():
            if not meteor.node:
                continue

            world_pos = meteor.node.getPos()
            is_embedded = (meteor.state == MeteorState.EMBEDDED)

            # Transform to camera space then to clip space
            cam_point = proj_mat.xformPoint(world_pos)
            clip = lens_mat.xform(Vec4(cam_point.x, cam_point.y, cam_point.z, 1.0))

            if clip.w <= 0.01:
                # Behind camera — place arrow based on direction
                sx, sy = cam_point.x, cam_point.z
                norm = max(abs(sx), abs(sy), 0.01)
                sx, sy = sx / norm * 2.0, sy / norm * 2.0
                on_screen = False
            else:
                ndc_x = clip.x / clip.w
                ndc_y = clip.z / clip.w
                # Panda3D aspect2d range: x ~ -1.33..1.33, y ~ -1..1
                aspect = self.getAspectRatio() if hasattr(self, 'getAspectRatio') else 1.333
                sx = ndc_x * aspect
                sy = ndc_y
                on_screen = (-1.2 < sx < 1.2) and (-0.9 < sy < 0.9)

            dist = (world_pos - self._player.position).length()
            meteor_data.append((sx, sy, on_screen, is_embedded, dist))

        self._hud.update_arrows(meteor_data)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    app = OrbitalGuardian()
    app.run()
