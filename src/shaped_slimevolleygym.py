from typing import Any

import numpy as np
from gymnasium.envs.registration import register

from slimevolleygym.slimevolley import SlimeVolleyEnv


class ShapedSlimeVolleyEnv(SlimeVolleyEnv):
    """
    Custom wrapper class that inherits from SlimeVolleyEnv.
    Fails the rally and penalizes an agent if it juggles/touches
    the ball more than 3 times without it crossing the net.
    Strict milestone shaping: Rewards for point win/loss and fast net crossings.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_touches: int = 3
        self._reset_touch_state()

    def _reset_touch_state(self) -> None:
        """Helper to cleanly reset internal tracking states."""
        self.right_touches: int = 0
        self.left_touches: int = 0
        self.prev_bx: float = 0.0

        # State transition flags for spatial collisions
        self.was_touching_right: bool = False
        self.was_touching_left: bool = False

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self._reset_touch_state()
        return super().reset(**kwargs)

    def step(
        self, action: np.ndarray, otherAction: Any = None
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

        obs, base_reward, terminated, truncated, info = super().step(
            action, otherAction
        )

        reward = float(base_reward)

        # 4: ball_x, 6: ball_vx
        bx = obs[4]
        bvx = obs[6]

        # End of rally - skip shaping and reset states for the serve
        if base_reward != 0:
            self._reset_touch_state()
            return obs, reward, terminated, truncated, info

        # --- MILESTONE REWARD SHAPING ---

        # Net cross reward
        if (self.prev_bx > 0 > bx) or (self.prev_bx < 0 < bx):
            base_cross_reward = 0.01
            # This rewards aggressive drives and spikes.
            velocity_bonus = abs(bvx) * 0.0
            reward += base_cross_reward + velocity_bonus

        self.prev_bx = bx

        # --- TOUCH TRACKING & RULES ---

        # Reset touches when ball changes sides
        if bx > 0:
            self.left_touches = 0
        elif bx < 0:
            self.right_touches = 0

        # Bounce / Touch detection via buffered spatial collision
        ball = self.game.ball
        p_right = self.game.agent_right
        p_left = self.game.agent_left

        # Calculate squared distances
        dist_sq_right = (ball.x - p_right.x) ** 2 + (ball.y - p_right.y) ** 2
        dist_sq_left = (ball.x - p_left.x) ** 2 + (ball.y - p_left.y) ** 2

        # The Physics Buffer:
        BUFFER = ball.r * 0.25
        col_dist_sq_right = (p_right.r + ball.r + BUFFER) ** 2
        col_dist_sq_left = (p_left.r + ball.r + BUFFER) ** 2

        # Check current physical overlap (within the buffer zone)
        is_touching_right = dist_sq_right <= col_dist_sq_right
        is_touching_left = dist_sq_left <= col_dist_sq_left

        # Only count if touching NOW, but was NOT touching last frame
        if is_touching_right and not self.was_touching_right:
            self.right_touches += 1

        if is_touching_left and not self.was_touching_left:
            self.left_touches += 1

        # Save the overlap state for the next frame
        self.was_touching_right = is_touching_right
        self.was_touching_left = is_touching_left

        # Enforce 3-touch rule for Right Agent
        if self.right_touches > self.max_touches:
            reward = -1.0
            self.right_touches = 0
            self.left_touches = 0
            self.game.agent_right.life -= 1
            self.game.agent_left.emotion = "happy"
            self.game.agent_right.emotion = "sad"
            self.game.newMatch()

            if self.game.agent_right.life <= 0:
                terminated = True

        # Enforce 3-touch rule for Left Agent
        elif self.left_touches > self.max_touches:
            reward = 1.0
            self.right_touches = 0
            self.left_touches = 0
            self.game.agent_left.life -= 1
            self.game.agent_left.emotion = "sad"
            self.game.agent_right.emotion = "happy"
            self.game.newMatch()

            if self.game.agent_left.life <= 0:
                terminated = True

        return obs, reward, terminated, truncated, info


# Register the shaped environment
register(
    id="SlimeVolleyShaped-v0",
    entry_point="shaped_slimevolleygym:ShapedSlimeVolleyEnv",
)
