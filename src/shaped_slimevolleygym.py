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
        self.was_moving_down: bool = False
        self.touch_cooldown: int = 0
        self.prev_bx: float = 0.0

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

        # 4: ball_x, 6: ball_vx, 7: ball_vy
        bx = obs[4]
        bvx = obs[6]
        bvy = obs[7]

        if self.touch_cooldown > 0:
            self.touch_cooldown -= 1

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
        is_moving_down = bvy < 0

        # Reset touches when ball changes sides
        if bx > 0:
            self.left_touches = 0
        elif bx < 0:
            self.right_touches = 0

        # Bounce / Touch detection via velocity shift
        if self.was_moving_down and not is_moving_down and self.touch_cooldown == 0:
            self.touch_cooldown = 5

            if bx > 0:
                self.right_touches += 1
            elif bx < 0:
                self.left_touches += 1

        self.was_moving_down = is_moving_down

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
