"""
Port of Neural Slime Volleyball to Python Gymnasium Environment

David Ha (2020) – original gym version
Modernized to Gymnasium API

Original:
  https://otoro.net/slimevolley
  https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
  https://github.com/hardmaru/neuralslimevolley

Dependencies: gymnasium, numpy, opencv-python, pygame (human rendering only)
"""

import math
from collections import deque

import cv2
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# ──────────────────────────── Game constants ────────────────────────────────
REF_W = 24 * 2
REF_H = REF_W
REF_U = 1.5  # ground height
REF_WALL_WIDTH = 1.0  # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10 * 1.75
PLAYER_SPEED_Y = 10 * 1.35
MAX_BALL_SPEED = 15 * 1.5
TIMESTEP = 1 / 30.0
NUDGE = 0.1
FRICTION = 1.0  # 1 = no friction
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8 * 2 * 1.5
MAXLIVES = 5  # game ends when one agent loses this many lives

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500
FACTOR = WINDOW_WIDTH / REF_W

PIXEL_SCALE = 4  # render at 4× pixel-obs resolution, then downscale
PIXEL_WIDTH = 84 * 2  # pixel observation width
PIXEL_HEIGHT = 84  # pixel observation height


# ──────────────────────────── Color themes ──────────────────────────────────
def setNightColors():
    global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
    global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
    global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
    BALL_COLOR = (217, 79, 0)
    AGENT_LEFT_COLOR = (35, 93, 188)
    AGENT_RIGHT_COLOR = (255, 236, 0)
    PIXEL_AGENT_LEFT_COLOR = (255, 191, 0)  # amber – same for both in pixel mode
    PIXEL_AGENT_RIGHT_COLOR = (255, 191, 0)
    BACKGROUND_COLOR = (11, 16, 19)
    FENCE_COLOR = (102, 56, 35)
    COIN_COLOR = FENCE_COLOR
    GROUND_COLOR = (116, 114, 117)


def setDayColors():
    global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
    global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
    global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
    BALL_COLOR = (255, 200, 20)
    AGENT_LEFT_COLOR = (240, 75, 0)
    AGENT_RIGHT_COLOR = (0, 150, 255)
    PIXEL_AGENT_LEFT_COLOR = (240, 75, 0)
    PIXEL_AGENT_RIGHT_COLOR = (0, 150, 255)
    BACKGROUND_COLOR = (255, 255, 255)
    FENCE_COLOR = (240, 210, 130)
    COIN_COLOR = FENCE_COLOR
    GROUND_COLOR = (128, 227, 153)


setNightColors()  # default theme


# ──────────────────────────── Pixel-obs mode ────────────────────────────────
def setPixelObsMode():
    """Switch global rendering dimensions to pixel-observation mode.

    Note: this modifies module-level globals, so only one pixel env at a time
    is supported if you mix state-obs and pixel-obs environments.
    """
    global WINDOW_WIDTH, WINDOW_HEIGHT, FACTOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
    WINDOW_WIDTH = PIXEL_WIDTH * PIXEL_SCALE
    WINDOW_HEIGHT = PIXEL_HEIGHT * PIXEL_SCALE
    FACTOR = WINDOW_WIDTH / REF_W
    AGENT_LEFT_COLOR = PIXEL_AGENT_LEFT_COLOR
    AGENT_RIGHT_COLOR = PIXEL_AGENT_RIGHT_COLOR


def upsize_image(img):
    return cv2.resize(
        img,
        (PIXEL_WIDTH * PIXEL_SCALE, PIXEL_HEIGHT * PIXEL_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )


def downsize_image(img):
    return cv2.resize(img, (PIXEL_WIDTH, PIXEL_HEIGHT), interpolation=cv2.INTER_AREA)


# ──────────────────────────── Coordinate transforms ─────────────────────────
def toX(x):
    return (x + REF_W / 2) * FACTOR


def toP(x):
    return x * FACTOR


def toY(y):
    return y * FACTOR


# ──────────────────────────── Delay screen ──────────────────────────────────
class DelayScreen:
    """Hold the ball still for INIT_DELAY_FRAMES at the start of each match."""

    def __init__(self, life=INIT_DELAY_FRAMES):
        self.life = 0
        self.reset(life)

    def reset(self, life=INIT_DELAY_FRAMES):
        self.life = life

    def status(self):
        if self.life == 0:
            return True
        self.life -= 1
        return False


# ──────────────────────────── Drawing helpers (cv2/numpy) ───────────────────
# All rendering goes through cv2 into an RGB numpy array.
# Colors throughout are (R, G, B) tuples; pygame and the obs space both use RGB.


def create_canvas(canvas, c):
    """Return a fresh (WINDOW_HEIGHT, WINDOW_WIDTH, 3) canvas filled with color c."""
    result = np.empty((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    result[:] = c
    return result


def rect(canvas, x, y, width, height, color):
    h = canvas.shape[0]
    return cv2.rectangle(
        canvas,
        (round(x), round(h - y)),
        (round(x + width), round(h - y + height)),
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def half_circle(canvas, x, y, r, color):
    h = canvas.shape[0]
    return cv2.ellipse(
        canvas,
        (round(x), h - round(y)),
        (round(r), round(r)),
        0,
        0,
        -180,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def circle(canvas, x, y, r, color):
    h = canvas.shape[0]
    return cv2.circle(
        canvas,
        (round(x), round(h - y)),
        max(1, round(r)),
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ──────────────────────────── Game objects ──────────────────────────────────
class Particle:
    """Used for the ball and the round stub above the fence."""

    def __init__(self, x, y, vx, vy, r, c):
        self.x = x
        self.y = y
        self.prev_x = self.x
        self.prev_y = self.y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.c = c

    def display(self, canvas):
        return circle(canvas, toX(self.x), toY(self.y), toP(self.r), color=self.c)

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def applyAcceleration(self, ax, ay):
        self.vx += ax * TIMESTEP
        self.vy += ay * TIMESTEP

    def checkEdges(self):
        if self.x <= self.r - REF_W / 2:
            self.vx *= -FRICTION
            self.x = self.r - REF_W / 2 + NUDGE * TIMESTEP

        if self.x >= REF_W / 2 - self.r:
            self.vx *= -FRICTION
            self.x = REF_W / 2 - self.r - NUDGE * TIMESTEP

        if self.y <= self.r + REF_U:
            self.vy *= -FRICTION
            self.y = self.r + REF_U + NUDGE * TIMESTEP
            return -1 if self.x <= 0 else 1

        if self.y >= REF_H - self.r:
            self.vy *= -FRICTION
            self.y = REF_H - self.r - NUDGE * TIMESTEP

        # Fence collisions
        if (
            self.x <= REF_WALL_WIDTH / 2 + self.r
            and self.prev_x > REF_WALL_WIDTH / 2 + self.r
            and self.y <= REF_WALL_HEIGHT
        ):
            self.vx *= -FRICTION
            self.x = REF_WALL_WIDTH / 2 + self.r + NUDGE * TIMESTEP

        if (
            self.x >= -REF_WALL_WIDTH / 2 - self.r
            and self.prev_x < -REF_WALL_WIDTH / 2 - self.r
            and self.y <= REF_WALL_HEIGHT
        ):
            self.vx *= -FRICTION
            self.x = -REF_WALL_WIDTH / 2 - self.r - NUDGE * TIMESTEP

        return 0

    def getDist2(self, p):
        dy = p.y - self.y
        dx = p.x - self.x
        return dx * dx + dy * dy

    def isColliding(self, p):
        r = self.r + p.r
        return r * r > self.getDist2(p)

    def bounce(self, p):
        abx = self.x - p.x
        aby = self.y - p.y
        abd = math.sqrt(abx * abx + aby * aby)
        abx /= abd
        aby /= abd
        nx, ny = abx, aby
        abx *= NUDGE
        aby *= NUDGE
        while self.isColliding(p):
            self.x += abx
            self.y += aby
        ux = self.vx - p.vx
        uy = self.vy - p.vy
        un = ux * nx + uy * ny
        ux -= nx * (un * 2.0)
        uy -= ny * (un * 2.0)
        self.vx = ux + p.vx
        self.vy = uy + p.vy

    def limitSpeed(self, minSpeed, maxSpeed):
        mag2 = self.vx * self.vx + self.vy * self.vy
        if mag2 > maxSpeed * maxSpeed:
            mag = math.sqrt(mag2)
            self.vx = self.vx / mag * maxSpeed
            self.vy = self.vy / mag * maxSpeed
        if mag2 < minSpeed * minSpeed and mag2 > 0:
            mag = math.sqrt(mag2)
            self.vx = self.vx / mag * minSpeed
            self.vy = self.vy / mag * minSpeed


class Wall:
    """Used for the fence and the ground."""

    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    def display(self, canvas):
        return rect(
            canvas,
            toX(self.x - self.w / 2),
            toY(self.y + self.h / 2),
            toP(self.w),
            toP(self.h),
            color=self.c,
        )


class RelativeState:
    """Observation from one agent's perspective (always from the right side)."""

    def __init__(self):
        self.x = self.y = self.vx = self.vy = 0.0
        self.bx = self.by = self.bvx = self.bvy = 0.0
        self.ox = self.oy = self.ovx = self.ovy = 0.0

    def getObservation(self):
        obs = [
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.bx,
            self.by,
            self.bvx,
            self.bvy,
            self.ox,
            self.oy,
            self.ovx,
            self.ovy,
        ]
        return np.array(obs, dtype=np.float32) / 10.0  # scale to ~order-of-magnitude 1


class Agent:
    """In-game agent state (not the policy network)."""

    def __init__(self, dir, x, y, c):
        self.dir = dir  # -1 = left player, +1 = right player
        self.x = x
        self.y = y
        self.r = 1.5
        self.c = c
        self.vx = self.vy = 0.0
        self.desired_vx = self.desired_vy = 0.0
        self.state = RelativeState()
        self.emotion = "happy"
        self.life = MAXLIVES

    def lives(self):
        return self.life

    def setAction(self, action):
        forward = action[0] > 0
        backward = action[1] > 0
        jump = action[2] > 0
        self.desired_vx = 0.0
        self.desired_vy = 0.0
        if forward and not backward:
            self.desired_vx = -PLAYER_SPEED_X
        if backward and not forward:
            self.desired_vx = PLAYER_SPEED_X
        if jump:
            self.desired_vy = PLAYER_SPEED_Y

    def move(self):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def step(self):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def update(self):
        self.vy += GRAVITY * TIMESTEP
        if self.y <= REF_U + NUDGE * TIMESTEP:
            self.vy = self.desired_vy
        self.vx = self.desired_vx * self.dir
        self.move()
        if self.y <= REF_U:
            self.y = REF_U
            self.vy = 0.0
        # Stay in own half
        if self.x * self.dir <= REF_WALL_WIDTH / 2 + self.r:
            self.vx = 0.0
            self.x = self.dir * (REF_WALL_WIDTH / 2 + self.r)
        if self.x * self.dir >= REF_W / 2 - self.r:
            self.vx = 0.0
            self.x = self.dir * (REF_W / 2 - self.r)

    def updateState(self, ball, opponent):
        """Normalise observation so both agents see the world from the right side."""
        self.state.x = self.x * self.dir
        self.state.y = self.y
        self.state.vx = self.vx * self.dir
        self.state.vy = self.vy
        self.state.bx = ball.x * self.dir
        self.state.by = ball.y
        self.state.bvx = ball.vx * self.dir
        self.state.bvy = ball.vy
        self.state.ox = opponent.x * (-self.dir)
        self.state.oy = opponent.y
        self.state.ovx = opponent.vx * (-self.dir)
        self.state.ovy = opponent.vy

    def getObservation(self):
        return self.state.getObservation()

    def display(self, canvas, bx, by):
        x, y, r = self.x, self.y, self.r
        angle = math.pi * 60 / 180 if self.dir == -1 else math.pi * 120 / 180
        canvas = half_circle(canvas, toX(x), toY(y), toP(r), color=self.c)

        c_a = math.cos(angle)
        s_a = math.sin(angle)
        ballX = bx - (x + 0.6 * r * c_a)
        ballY = by - (y + 0.6 * r * s_a)

        if self.emotion == "sad":
            ballX = -self.dir
            ballY = -3.0

        dist = math.sqrt(ballX * ballX + ballY * ballY)
        eyeX = ballX / dist
        eyeY = ballY / dist

        canvas = circle(
            canvas,
            toX(x + 0.6 * r * c_a),
            toY(y + 0.6 * r * s_a),
            toP(r) * 0.3,
            color=(255, 255, 255),
        )
        canvas = circle(
            canvas,
            toX(x + 0.6 * r * c_a + eyeX * 0.15 * r),
            toY(y + 0.6 * r * s_a + eyeY * 0.15 * r),
            toP(r) * 0.1,
            color=(0, 0, 0),
        )

        # Draw remaining lives as coins along the top edge
        for i in range(1, self.life):
            canvas = circle(
                canvas,
                toX(self.dir * (REF_W / 2 + 0.5 - i * 2.0)),
                WINDOW_HEIGHT - toY(1.5),
                toP(0.5),
                color=COIN_COLOR,
            )
        return canvas


class BaselinePolicy:
    """Tiny recurrent policy (120 parameters) from otoro.net/slimevolley."""

    def __init__(self):
        self.nGameInput = 8
        self.nGameOutput = 3
        self.nRecurrentState = 4
        self.nOutput = self.nGameOutput + self.nRecurrentState
        self.nInput = self.nGameInput + self.nOutput

        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

        # Weights from the original blog post:
        # https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
        self.weight = np.array(
            [
                7.5719,
                4.4285,
                2.2716,
                -0.3598,
                -7.8189,
                -2.5422,
                -3.2034,
                0.3935,
                1.2202,
                -0.49,
                -0.0316,
                0.5221,
                0.7026,
                0.4179,
                -2.1689,
                1.646,
                -13.3639,
                1.5151,
                1.1175,
                -5.3561,
                5.0442,
                0.8451,
                0.3987,
                -2.9501,
                -3.7811,
                -5.8994,
                6.4167,
                2.5014,
                7.338,
                -2.9887,
                2.4586,
                13.4191,
                2.7395,
                -3.9708,
                1.6548,
                -2.7554,
                -1.5345,
                -6.4708,
                9.2426,
                -0.7392,
                0.4452,
                1.8828,
                -2.6277,
                -10.851,
                -3.2353,
                -4.4653,
                -3.1153,
                -1.3707,
                7.318,
                16.0902,
                1.4686,
                7.0391,
                1.7765,
                -1.155,
                2.6697,
                -8.8877,
                1.1958,
                -3.2839,
                -5.4425,
                1.6809,
                7.6812,
                -2.4732,
                1.738,
                0.3781,
                0.8718,
                2.5886,
                1.6911,
                1.2953,
                -9.0052,
                -4.6038,
                -6.7447,
                -2.5528,
                0.4391,
                -4.9278,
                -3.6695,
                -4.8673,
                -1.6035,
                1.5011,
                -5.6124,
                4.9747,
                1.8998,
                3.0359,
                6.2983,
                -4.8568,
                -2.1888,
                -4.1143,
                -3.9874,
                -0.0459,
                4.7134,
                2.8952,
                -9.3627,
                -4.685,
                0.3601,
                -1.3699,
                9.7294,
                11.5596,
                0.1918,
                3.0783,
                0.0329,
                -0.1362,
                -0.1188,
                -0.7579,
                0.3278,
                -0.977,
                -0.9377,
            ]
        )
        self.bias = np.array(
            [2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685]
        )
        self.weight = self.weight.reshape(
            self.nGameOutput + self.nRecurrentState,
            self.nGameInput + self.nGameOutput + self.nRecurrentState,
        )

    def reset(self):
        self.inputState[:] = 0.0
        self.outputState[:] = 0.0
        self.prevOutputState[:] = 0.0

    def _forward(self):
        self.prevOutputState = self.outputState
        self.outputState = np.tanh(np.dot(self.weight, self.inputState) + self.bias)

    def _setInputState(self, obs):
        [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
        self.inputState[: self.nGameInput] = [
            x,
            y,
            vx,
            vy,
            ball_x,
            ball_y,
            ball_vx,
            ball_vy,
        ]
        self.inputState[self.nGameInput :] = self.outputState

    def _getAction(self):
        return [
            int(self.outputState[0] > 0.75),
            int(self.outputState[1] > 0.75),
            int(self.outputState[2] > 0.75),
        ]

    def predict(self, obs):
        """Take obs, update RNN state, return action."""
        self._setInputState(obs)
        self._forward()
        return self._getAction()


class Game:
    """Core slime volleyball physics and game logic."""

    def __init__(self, np_random=None):
        self.np_random = np_random if np_random is not None else np.random.default_rng()
        # Initialise all game objects via reset() so type annotations are satisfied.
        self.reset()

    def reset(self):
        self.ground = Wall(0, 0.75, REF_W, REF_U, c=GROUND_COLOR)
        self.fence = Wall(
            0,
            0.75 + REF_WALL_HEIGHT / 2,
            REF_WALL_WIDTH,
            REF_WALL_HEIGHT - 1.5,
            c=FENCE_COLOR,
        )
        self.fenceStub = Particle(
            0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2, c=FENCE_COLOR
        )
        ball_vx = float(self.np_random.uniform(-20, 20))
        ball_vy = float(self.np_random.uniform(10, 25))
        self.ball = Particle(0, REF_W / 4, ball_vx, ball_vy, 0.5, c=BALL_COLOR)
        self.agent_left = Agent(-1, -REF_W / 4, 1.5, c=AGENT_LEFT_COLOR)
        self.agent_right = Agent(1, REF_W / 4, 1.5, c=AGENT_RIGHT_COLOR)
        self.agent_left.updateState(self.ball, self.agent_right)
        self.agent_right.updateState(self.ball, self.agent_left)
        self.delayScreen = DelayScreen()

    def newMatch(self):
        ball_vx = float(self.np_random.uniform(-20, 20))
        ball_vy = float(self.np_random.uniform(10, 25))
        self.ball = Particle(0, REF_W / 4, ball_vx, ball_vy, 0.5, c=BALL_COLOR)
        self.delayScreen.reset()

    def step(self):
        """Advance one timestep. Returns reward from the right agent's perspective."""
        self.betweenGameControl()
        self.agent_left.update()
        self.agent_right.update()

        if self.delayScreen.status():
            self.ball.applyAcceleration(0, GRAVITY)
            self.ball.limitSpeed(0, MAX_BALL_SPEED)
            self.ball.move()

        if self.ball.isColliding(self.agent_left):
            self.ball.bounce(self.agent_left)
        if self.ball.isColliding(self.agent_right):
            self.ball.bounce(self.agent_right)
        if self.ball.isColliding(self.fenceStub):
            self.ball.bounce(self.fenceStub)

        result = -self.ball.checkEdges()  # negate: right agent wins = +1

        if result != 0:
            self.newMatch()
            if result < 0:  # left agent scored
                self.agent_left.emotion = "happy"
                self.agent_right.emotion = "sad"
                self.agent_right.life -= 1
            else:  # right agent scored
                self.agent_left.emotion = "sad"
                self.agent_right.emotion = "happy"
                self.agent_left.life -= 1
            return result

        self.agent_left.updateState(self.ball, self.agent_right)
        self.agent_right.updateState(self.ball, self.agent_left)
        return result

    def display(self, canvas):
        """Render the current game state to an RGB numpy array."""
        canvas = create_canvas(canvas, c=BACKGROUND_COLOR)
        canvas = self.fence.display(canvas)
        canvas = self.fenceStub.display(canvas)
        canvas = self.agent_left.display(canvas, self.ball.x, self.ball.y)
        canvas = self.agent_right.display(canvas, self.ball.x, self.ball.y)
        canvas = self.ball.display(canvas)
        canvas = self.ground.display(canvas)
        return canvas

    def betweenGameControl(self):
        if self.delayScreen.life <= 0:
            self.agent_left.emotion = "happy"
            self.agent_right.emotion = "happy"


# ──────────────────────────── Gymnasium environment ─────────────────────────
class SlimeVolleyEnv(gymnasium.Env):
    """
    Gymnasium environment for Slime Volleyball.

    The **right** agent is the one being trained; the **left** agent defaults
    to the built-in 120-parameter RNN baseline policy.

    Single-agent usage (standard Gymnasium loop):

        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    Multi-agent / self-play usage (pass a second action to step):

        obs, reward, terminated, truncated, info = env.step(action1, action2)
        obs2 = info['otherObs']   # left agent's observation

    Reward is from the **right** agent's perspective (+1 / -1 per life lost).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # Atari-style action meanings (kept for compatibility)
    atari_action_meaning = {
        0: "NOOP",
        1: "FIRE",
        2: "UP",
        3: "RIGHT",
        4: "LEFT",
        5: "DOWN",
        6: "UPRIGHT",
        7: "UPLEFT",
        8: "DOWNRIGHT",
        9: "DOWNLEFT",
        10: "UPFIRE",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
        13: "DOWNFIRE",
        14: "UPRIGHTFIRE",
        15: "UPLEFTFIRE",
        16: "DOWNRIGHTFIRE",
        17: "DOWNLEFTFIRE",
    }
    atari_action_set = {0, 4, 7, 2, 6, 3}

    action_table = [
        [0, 0, 0],  # NOOP
        [1, 0, 0],  # LEFT (forward)
        [1, 0, 1],  # UPLEFT (forward + jump)
        [0, 0, 1],  # UP (jump)
        [0, 1, 1],  # UPRIGHT (backward + jump)
        [0, 1, 0],  # RIGHT (backward)
    ]

    from_pixels = False
    atari_mode = False
    survival_bonus = False
    multiagent = True

    def __init__(self, render_mode=None):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"render_mode must be one of {self.metadata['render_modes']!r}, got {render_mode!r}"
        )
        self.render_mode = render_mode
        self.t = 0
        self.t_limit = 3000

        if self.atari_mode:
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.MultiBinary(3)

        if self.from_pixels:
            setPixelObsMode()
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3),
                dtype=np.uint8,
            )
        else:
            high = np.full(12, np.finfo(np.float32).max, dtype=np.float32)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.game = Game()
        self.ale = (
            self.game.agent_right
        )  # compatibility shim for models using ale.lives()
        self.policy = BaselinePolicy()
        self.otherAction = None  # can be set externally to override left agent

        # Pygame window (lazy-initialised on first human render)
        self._window = None
        self._clock = None

    # ── Observation helpers ──────────────────────────────────────────────────
    def getObs(self):
        if self.from_pixels:
            canvas = self.game.display(None)
            return downsize_image(canvas)
        return self.game.agent_right.getObservation()

    def discreteToBox(self, n):
        """Convert a discrete Atari-style action integer to the 3-bit action tuple."""
        if isinstance(n, (list, tuple, np.ndarray)):
            if len(n) == 3:
                return n
        n = int(n)  # type: ignore[arg-type]  – narrowed by isinstance above
        assert 0 <= n < 6, f"Discrete action must be in [0, 5], got {n}"
        return self.action_table[n]

    # ── Gymnasium API ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Propagate numpy Generator into game for deterministic resets
        self.game.np_random = self.np_random
        self.t = 0
        self.game.reset()
        self.ale = self.game.agent_right  # refresh after game.reset() creates new Agent
        self.policy.reset()
        obs = self.getObs()
        info = {
            "otherObs": self.game.agent_left.getObservation(),
        }
        return obs, info

    def step(self, action, otherAction=None):
        """
        Step the environment.

        Args:
            action:      Action for the right (training) agent.
            otherAction: Optional action for the left agent.  When ``None``,
                         the built-in baseline policy is used.

        Returns:
            ``(obs, reward, terminated, truncated, info)``
        """
        self.t += 1

        # Resolve left agent action
        if self.otherAction is not None:
            otherAction = self.otherAction
        if otherAction is None:
            otherAction = self.policy.predict(self.game.agent_left.getObservation())

        if self.atari_mode:
            action = self.discreteToBox(action)
            otherAction = self.discreteToBox(otherAction)

        self.game.agent_left.setAction(otherAction)
        self.game.agent_right.setAction(action)
        reward = float(self.game.step())
        obs = self.getObs()

        terminated = self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0
        truncated = self.t >= self.t_limit

        otherObs = None
        if self.multiagent:
            otherObs = (
                cv2.flip(obs, 1)
                if self.from_pixels
                else self.game.agent_left.getObservation()
            )

        info = {
            "ale.lives": self.game.agent_right.lives(),
            "ale.otherLives": self.game.agent_left.lives(),
            "otherObs": otherObs,
            "state": self.game.agent_right.getObservation(),
            "otherState": self.game.agent_left.getObservation(),
        }

        if self.survival_bonus:
            reward += 0.01

        return obs, reward, terminated, truncated, info

    # ── Rendering ────────────────────────────────────────────────────────────
    def render(self):
        """Render the current frame.

        Returns an RGB numpy array when ``render_mode='rgb_array'``, otherwise
        None (the frame is shown in a pygame window).
        """
        if self.render_mode is None:
            return

        canvas = self.game.display(None)  # (WINDOW_HEIGHT, WINDOW_WIDTH, 3) RGB array

        if self.render_mode == "rgb_array":
            if self.from_pixels:
                return downsize_image(canvas)
            return canvas.copy()

        # ── human mode: display via pygame ──────────────────────────────────
        import pygame  # optional dependency; only imported when rendering

        if self._window is None:
            pygame.init()
            pygame.display.init()
            h, w = canvas.shape[:2]
            self._window = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Slime Volleyball")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # numpy array is (H, W, 3) RGB; pygame surfarray wants (W, H, 3)
        surf = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
        self._window.blit(surf, (0, 0))
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None

    def get_action_meanings(self):
        return [self.atari_action_meaning[i] for i in self.atari_action_set]


# ──────────────────────────── Env variants ──────────────────────────────────
class SlimeVolleyPixelEnv(SlimeVolleyEnv):
    """Pixel-observation variant: obs is (84, 168, 3) RGB."""

    from_pixels = True


class SlimeVolleyAtariEnv(SlimeVolleyEnv):
    """Pixel + Discrete(6) action space, matching typical Atari wrappers."""

    from_pixels = True
    atari_mode = True


class SlimeVolleySurvivalAtariEnv(SlimeVolleyEnv):
    """Atari variant with a small per-step survival bonus (+0.01)."""

    from_pixels = True
    atari_mode = True
    survival_bonus = True


# ──────────────────────────── Wrappers ──────────────────────────────────────
class SurvivalRewardEnv(gymnasium.RewardWrapper):
    """Adds 0.01 survival bonus to every timestep reward."""

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return float(reward) + 0.01


class FrameStack(gymnasium.Wrapper):
    """Stack the last *n_frames* observations along the channel axis."""

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return np.concatenate(list(self.frames), axis=2)


# ──────────────────────────── Helper functions ──────────────────────────────
def multiagent_rollout(env, policy_right, policy_left, render_mode=False):
    """
    Run one full episode with two policies.

    Returns:
        (total_reward, timesteps) where total_reward is from *policy_right*'s
        perspective (positive = right agent winning).
    """
    obs_right, info = env.reset()
    obs_left = obs_right  # both sides share the same initial observation
    done = False
    total_reward = 0
    t = 0

    while not done:
        action_right = policy_right.predict(obs_right)
        action_left = policy_left.predict(obs_left)

        obs_right, reward, terminated, truncated, info = env.step(
            action_right, action_left
        )
        obs_left = info["otherObs"]
        done = terminated or truncated
        total_reward += reward
        t += 1

        if render_mode:
            env.render()

    return total_reward, t


def render_atari(obs):
    """Visualise a stacked Atari observation of shape (84, 84, 4).

    Returns an RGB image with the latest frame (top) and all 4 frames (bottom).
    """
    obs = np.copy(obs)
    frames = []
    for i in range(4):
        if i > 0:
            obs[:, 0, i] = 141  # separator line
        frames.append(obs[:, :, i])

    latest = np.expand_dims(frames[-1], axis=2)
    latest = np.concatenate([latest * 255.0] * 3, axis=2).astype(np.uint8)
    latest = cv2.resize(latest, (84 * 8, 84 * 4), interpolation=cv2.INTER_NEAREST)

    strip = np.expand_dims(np.concatenate(frames, axis=1), axis=2)
    strip = np.concatenate([strip * 255.0] * 3, axis=2).astype(np.uint8)
    strip = cv2.resize(strip, (84 * 8, 84 * 2), interpolation=cv2.INTER_NEAREST)

    return np.concatenate([latest, strip], axis=0)


# ──────────────────────────── Register environments ─────────────────────────
register(
    id="SlimeVolley-v0",
    entry_point="slimevolleygym.slimevolley:SlimeVolleyEnv",
)
register(
    id="SlimeVolleyPixel-v0",
    entry_point="slimevolleygym.slimevolley:SlimeVolleyPixelEnv",
)
register(
    id="SlimeVolleyNoFrameskip-v0",
    entry_point="slimevolleygym.slimevolley:SlimeVolleyAtariEnv",
)
register(
    id="SlimeVolleySurvivalNoFrameskip-v0",
    entry_point="slimevolleygym.slimevolley:SlimeVolleySurvivalAtariEnv",
)
