"""
example_agents.py
=================
Two BaselinePolicy agents play against each other in SlimeVolley-v0.

This script verifies:
  - The gymnasium API (reset returns (obs, info); step returns 5-tuple)
  - Multi-agent step (passing a second action)
  - info['otherObs'] for the left agent's observation
  - Human rendering via pygame (render_mode="human")
  - Proper episode termination (terminated / truncated flags)

Run with visualisation (default):
    python example_agents.py

Run headless (no window, prints stats only):
    python example_agents.py --no-render

Run multiple episodes:
    python example_agents.py --episodes 5

Notes
-----
The multi-agent step `env.step(action1, action2)` requires the *unwrapped*
environment because gymnasium's OrderEnforcing wrapper only accepts a single
action argument.

Two ways to do multi-agent steps:

    Option A – direct instantiation (used here):
        from slimevolleygym import SlimeVolleyEnv
        env = SlimeVolleyEnv(render_mode="human")

    Option B – gymnasium.make() + unwrapped attribute:
        env = gymnasium.make("SlimeVolley-v0", render_mode="human")
        env.unwrapped.otherAction = action_left
        obs, reward, terminated, truncated, info = env.step(action_right)
        env.unwrapped.otherAction = None   # reset so baseline is used next step
"""

import argparse
import time

import gymnasium
import numpy as np

import slimevolleygym  # noqa: F401 – registers the environments
from slimevolleygym import BaselinePolicy, SlimeVolleyEnv


def run_episode(
    env: SlimeVolleyEnv,
    policy_right: BaselinePolicy,
    policy_left: BaselinePolicy,
    render: bool,
) -> tuple[float, int]:
    """Run one episode and return (total_reward, num_steps)."""
    obs_right, info = env.reset()
    obs_left = obs_right  # both sides share the same first observation

    policy_right.reset()
    policy_left.reset()

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action_right = policy_right.predict(obs_right)
        action_left = policy_left.predict(obs_left)

        # Pass both actions directly – works because we use the unwrapped env.
        obs_right, reward, terminated, truncated, info = env.step(
            action_right, action_left
        )
        obs_left = info["otherObs"]  # left agent's observation (right-side perspective)

        done = terminated or truncated
        total_reward += reward
        steps += 1

        if render:
            env.render()

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(
        description="Slime Volleyball – baseline vs baseline"
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument(
        "--no-render", dest="render", action="store_false", help="Disable pygame window"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for first episode"
    )
    parser.set_defaults(render=True)
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    # ── API smoke test (through gymnasium.make so wrappers are exercised) ───
    print("Running API checks via gymnasium.make()…")
    wrapped_env = gymnasium.make("SlimeVolley-v0")  # no render needed for checks

    obs, info = wrapped_env.reset(seed=args.seed)
    assert isinstance(obs, np.ndarray), "obs should be a numpy array"
    assert obs.shape == (12,), f"Expected shape (12,), got {obs.shape}"
    assert isinstance(info, dict), "info should be a dict"
    print("✓ reset() returns (obs, info) with correct shapes")

    result = wrapped_env.step(wrapped_env.action_space.sample())
    assert len(result) == 5, f"step() should return 5 values, got {len(result)}"
    _, _, terminated, truncated, info2 = result
    assert isinstance(terminated, bool), "terminated must be bool"
    assert isinstance(truncated, bool), "truncated must be bool"
    assert "otherObs" in info2, "'otherObs' missing from info"
    print("✓ step()  returns (obs, reward, terminated, truncated, info)")
    wrapped_env.close()
    print()

    # ── Multi-agent episodes (direct env, no wrapper) ────────────────────────
    print(f"Starting {args.episodes} multi-agent episode(s)…")
    print(f"  Render mode : {render_mode or 'headless'}")
    print()

    # Directly instantiate so we can call step(action1, action2)
    env = SlimeVolleyEnv(render_mode=render_mode)

    policy_right = BaselinePolicy()  # right (training) agent
    policy_left = BaselinePolicy()  # left agent (opponent)

    scores: list[float] = []
    t0 = time.perf_counter()

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)  # re-seed each episode for reproducibility
        score, steps = run_episode(env, policy_right, policy_left, args.render)
        scores.append(score)
        winner = "right" if score > 0 else ("left" if score < 0 else "draw")
        print(
            f"Episode {ep + 1:3d} | steps: {steps:4d} | "
            f"score: {score:+.0f} | winner: {winner}"
        )

    elapsed = time.perf_counter() - t0
    print()
    print("─" * 45)
    print(f"Episodes  : {args.episodes}")
    print(f"Avg score : {np.mean(scores):+.3f}  (right agent perspective)")
    print(f"Std score : {np.std(scores):.3f}")
    print(f"Elapsed   : {elapsed:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
