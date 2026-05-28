"""
train_shaped.py — reward-shaped CEM training on SlimeVolley-v0.

Reward modifications over the sparse ±1/life signal:
  +BALL_TOUCH  when the right agent touches the ball
               (dense positioning signal; bounce detected *before* game.step()
                because game.step() moves the ball away after the bounce)
  -SHAKE       when the agent reverses its lateral direction in consecutive steps
               (discourages the aimless left-right oscillation untrained agents settle into)

Algorithm: Cross-Entropy Method (numpy-only, no ML framework).
With shaped rewards the agent starts developing useful behaviour in ~20 iterations
(~400 episodes, < 30 seconds).

Usage:
    uv run src/train_shaped.py                 # 20 iters × 20 pop
    uv run src/train_shaped.py --iters 40      # more training
    uv run src/train_shaped.py --no-vis
"""

from __future__ import annotations

import argparse

import gymnasium
import numpy as np
from tqdm import trange

from slimevolleygym import BaselinePolicy, SlimeVolleyEnv

# ── Reward constants ──────────────────────────────────────────────────────────
BALL_TOUCH_BONUS = 0.2  # agent touches ball
SHAKE_PENALTY = 0.05  # lateral direction reversal

# ── Policy architecture ───────────────────────────────────────────────────────
LAYER_SIZES = [12, 32, 3]  # obs → hidden → MultiBinary(3) actions

# ── CEM hyperparameters ───────────────────────────────────────────────────────
ITERS = 20
POP_SIZE = 20
ELITE_FRAC = 0.25


# ── Reward-shaping wrapper ────────────────────────────────────────────────────
class ShapedSlimeEnv(gymnasium.Wrapper):
    """Wraps SlimeVolleyEnv with ball-touch bonus and anti-shake penalty.

    Must directly wrap a SlimeVolleyEnv instance so that self.env.game is
    accessible for reading ball/agent state between physics steps.
    """

    def reset(self, **kwargs):
        self._prev_x = 0  # last non-zero lateral direction: +1 fwd / -1 bwd
        return self.env.reset(**kwargs)

    def step(self, action, otherAction=None):
        # ── Ball touch bonus ─────────────────────────────────────────────────
        # Check BEFORE game.step(): bounce() moves the ball away, so
        # isColliding() is False by the time we return from env.step().
        ball_touched = self.env.game.ball.isColliding(self.env.game.agent_right)

        obs, reward, terminated, truncated, info = self.env.step(action, otherAction)

        if ball_touched:
            reward += BALL_TOUCH_BONUS

        # ── Oscillation / shake penalty ──────────────────────────────────────
        a = np.asarray(action)
        cur_x = int(a[0] > 0) - int(a[1] > 0)  # +1 fwd, -1 bwd, 0 still

        if self._prev_x != 0 and cur_x != 0 and cur_x != self._prev_x:
            reward -= SHAKE_PENALTY  # direction reversed

        if cur_x != 0:
            self._prev_x = cur_x

        return obs, reward, terminated, truncated, info


# ── Numpy MLP policy ──────────────────────────────────────────────────────────
def n_params(sizes: list[int]) -> int:
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def forward(params: np.ndarray, obs: np.ndarray) -> list[int]:
    x, ptr = obs.astype(np.float32), 0
    for i in range(len(LAYER_SIZES) - 1):
        r, c = LAYER_SIZES[i], LAYER_SIZES[i + 1]
        W = params[ptr : ptr + r * c].reshape(r, c)
        ptr += r * c
        b = params[ptr : ptr + c]
        ptr += c
        x = np.tanh(x @ W + b)
    return (x > 0).astype(int).tolist()


# ── Evaluation (shaped env, baseline opponent) ────────────────────────────────
def evaluate(params: np.ndarray, n_episodes: int = 2) -> float:
    env = ShapedSlimeEnv(SlimeVolleyEnv())
    baseline = BaselinePolicy()
    total = 0.0
    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        obs_left = info["otherObs"]
        baseline.reset()
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(
                forward(params, obs), baseline.predict(obs_left)
            )
            obs_left = info["otherObs"]
            total += r
            done = term or trunc
    env.close()
    return total / n_episodes


# ── CEM training ──────────────────────────────────────────────────────────────
def train(n_iters: int, pop: int) -> np.ndarray:
    dim = n_params(LAYER_SIZES)
    n_elite = max(2, int(pop * ELITE_FRAC))
    mu = np.zeros(dim)
    sigma = np.ones(dim) * 0.5

    print(f"CEM + shaped rewards  |  iters={n_iters}  pop={pop}  params={dim}\n")
    print(
        f"  shaped reward = game_reward"
        f"  +{BALL_TOUCH_BONUS} (ball touch)"
        f"  -{SHAKE_PENALTY} (direction reversal)\n"
    )

    pbar = trange(n_iters, unit="iter")
    for _ in pbar:
        population = mu + sigma * np.random.randn(pop, dim)
        scores = [evaluate(p) for p in population]
        elite = population[np.argsort(scores)[-n_elite:]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-5
        pbar.set_postfix(best=f"{max(scores):+.2f}", mean=f"{np.mean(scores):+.2f}")

    return mu


# ── Visualisation (true game score, no shaping) ───────────────────────────────
def visualise(params: np.ndarray, n_episodes: int = 3) -> None:
    """Show the agent with UNMODIFIED game rules so score is directly comparable."""
    print(f"\nVisualising {n_episodes} episode(s) — true game score (no shaping).\n")
    env = SlimeVolleyEnv(render_mode="human")
    baseline = BaselinePolicy()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=200 + ep)
        obs_left = info["otherObs"]
        baseline.reset()
        done, score = False, 0.0

        while not done:
            action = forward(params, obs)
            obs, r, term, trunc, info = env.step(action, baseline.predict(obs_left))
            obs_left = info["otherObs"]
            score += float(r)
            done = term or trunc
            env.render()

        result = "win" if score > 0 else ("loss" if score < 0 else "draw")
        print(f"  episode {ep + 1}: score {score:+.0f}  [{result}]")

    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="CEM with shaped rewards on SlimeVolley-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--pop", type=int, default=POP_SIZE)
    p.add_argument("--vis-episodes", type=int, default=3)
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    best = train(args.iters, args.pop)

    if not args.no_vis:
        visualise(best, args.vis_episodes)


if __name__ == "__main__":
    main()
