from gymnasium.envs.registration import register

from slimevolleygym.slimevolley import SlimeVolleyEnv


class ShapedSlimeVolleyEnv(SlimeVolleyEnv):
    """
    Custom wrapper class that inherits from SlimeVolleyEnv.
    Fails the episode and penalizes an agent if it juggles/touches
    the ball more than 3 times without it crossing the net.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_touches = 3
        self.right_touches = 0
        self.left_touches = 0
        self.was_moving_down = False

        # Added a cooldown to prevent "ghost touches" from physics jitter
        self.touch_cooldown = 0

    def reset(self, **kwargs):
        self.right_touches = 0
        self.left_touches = 0
        self.was_moving_down = False
        self.touch_cooldown = 0
        return super().reset(**kwargs)

    def step(self, action, otherAction=None):
        obs, reward, terminated, truncated, info = super().step(action, otherAction)

        # Observation features mapping: 0:x, 1:y, 2:vx, 3:vy, 4:bx, 5:by, 6:bvx, 7:bvy
        bx = obs[4]
        bvy = obs[7]

        # Tick down the physics jitter cooldown
        if self.touch_cooldown > 0:
            self.touch_cooldown -= 1

        # If a normal point was just scored, reset touch counters for the new serve
        if reward != 0:
            self.right_touches = 0
            self.left_touches = 0
            self.was_moving_down = False
            self.touch_cooldown = 0
            return obs, reward, terminated, truncated, info

        is_moving_down = bvy < 0

        # Reset the opponent's touch count when the ball crosses the net
        if bx > 0:
            self.left_touches = 0
        elif bx < 0:
            self.right_touches = 0

        # Detect a bounce/touch (ball was moving down, now moving up/flat)
        # Added cooldown check to prevent rapid frame-by-frame ghost touches
        if self.was_moving_down and not is_moving_down and self.touch_cooldown == 0:
            self.touch_cooldown = 5  # Wait 5 frames before registering another touch

            if bx > 0:
                self.right_touches += 1
                # FIX 1: Only reward the FIRST touch to prevent juggling point-farming
                if self.right_touches == 1:
                    reward += 0.1
            elif bx < 0:
                self.left_touches += 1

        self.was_moving_down = is_moving_down

        # Enforce the 3-touch rule for the right agent (your model)
        if self.right_touches > self.max_touches:
            reward = -1.0  # Penalize the agent
            self.right_touches = 0
            self.left_touches = 0

            # FIX 2: Manually deduct a life and reset the rally instead of killing the episode
            self.game.agent_right.life -= 1
            self.game.agent_left.emotion = "happy"
            self.game.agent_right.emotion = "sad"
            self.game.newMatch()

            # Only terminate if the agent has actually run out of lives
            if self.game.agent_right.life <= 0:
                terminated = True

        # Enforce the 3-touch rule for the left agent (baseline opponent)
        elif self.left_touches > self.max_touches:
            reward = 1.0  # Your model gets a point
            self.right_touches = 0
            self.left_touches = 0

            # Deduct life from the opponent and reset rally
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
