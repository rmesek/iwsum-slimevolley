"""
train_ppo.py — train a PPO agent on SlimeVolley-v0, then visualise it.

The trained agent controls the RIGHT slime; the built-in baseline policy
controls the LEFT slime throughout training and visualisation.

Usage
-----
# Default: 50 000 training steps, then watch 3 episodes
uv run src/train_ppo.py

# Longer run (better agent, ~2-3 min on CPU)
uv run src/train_ppo.py --steps 300000

# Skip visualisation (headless)
uv run src/train_ppo.py --no-vis

# Load a previously saved model and just visualise
uv run src/train_ppo.py --steps 0 --model ppo_slimevolley
"""

import argparse
import time
from pathlib import Path

import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import slimevolleygym  # noqa: F401 – side-effect: registers env IDs
from slimevolleygym import BaselinePolicy, SlimeVolleyEnv

MODEL_DEFAULT = "ppo_slimevolley"


# ──────────────────────────── Training ─────────────────────────────────────
def train(total_timesteps: int, model_path: str) -> PPO:
    """Train a PPO agent against the built-in baseline policy."""
    print(f"\n{'=' * 55}")
    print(f" Training PPO  –  {total_timesteps:,} total timesteps")
    print(f"{'=' * 55}\n")
    print("The RIGHT agent is trained; the LEFT uses the 120-param baseline.")
    print("Expect a positive mean reward vs baseline after ~1 M steps.\n")

    # Standard single-agent env; baseline policy is handled internally.
    env = gymnasium.make("SlimeVolley-v0")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(model_path)
    env.close()

    print(f"\nModel saved → {model_path}.zip")
    return model


# ──────────────────────────── Headless evaluation ──────────────────────────
def evaluate(model: PPO, n_episodes: int = 5) -> float:
    """Run headless episodes and report the mean score vs baseline."""
    print(f"\n{'─' * 45}")
    print(f" Headless evaluation  ({n_episodes} episodes)")
    print(f"{'─' * 45}")

    # Direct instantiation needed for the multi-agent step API
    env = SlimeVolleyEnv(render_mode=None)
    baseline = BaselinePolicy()
    scores: list[float] = []

    for ep in range(n_episodes):
        obs_right, _ = env.reset(seed=ep)
        obs_left = obs_right
        baseline.reset()
        done = False
        total_reward = 0.0

        while not done:
            action_right, _ = model.predict(obs_right, deterministic=True)
            action_left = baseline.predict(obs_left)

            obs_right, reward, terminated, truncated, info = env.step(
                action_right, action_left
            )
            obs_left = info["otherObs"]
            total_reward += reward
            done = terminated or truncated

        scores.append(total_reward)
        result = "WIN" if total_reward > 0 else ("LOSS" if total_reward < 0 else "DRAW")
        print(f"  Episode {ep + 1:2d}:  score {total_reward:+.0f}  [{result}]")

    env.close()
    mean = float(np.mean(scores))
    print(f"\n  Mean score vs baseline: {mean:+.3f}")
    return mean


# ──────────────────────────── Visualisation ────────────────────────────────
def visualise(model: PPO, n_episodes: int = 3) -> None:
    """Render episodes in a pygame window: PPO (right) vs Baseline (left)."""
    print(f"\n{'=' * 55}")
    print(f" Visualisation — PPO (right, yellow) vs Baseline (left, blue)")
    print(f" {n_episodes} episode(s)   |   close window or Ctrl-C to quit")
    print(f"{'=' * 55}\n")

    env = SlimeVolleyEnv(render_mode="human")
    baseline = BaselinePolicy()

    try:
        for ep in range(n_episodes):
            obs_right, _ = env.reset(seed=200 + ep)
            obs_left = obs_right
            baseline.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action_right, _ = model.predict(obs_right, deterministic=True)
                action_left = baseline.predict(obs_left)

                obs_right, reward, terminated, truncated, info = env.step(
                    action_right, action_left
                )
                obs_left = info["otherObs"]
                total_reward += reward
                done = terminated or truncated
                steps += 1

                env.render()

            result = (
                "PPO"
                if total_reward > 0
                else ("Baseline" if total_reward < 0 else "Draw")
            )
            print(
                f"Episode {ep + 1}/{n_episodes}  "
                f"steps: {steps:4d}  score: {total_reward:+.0f}  "
                f"winner: {result}"
            )

            if ep < n_episodes - 1:
                time.sleep(1.0)  # brief pause between episodes

    except KeyboardInterrupt:
        print("\nVisualization interrupted.")
    finally:
        env.close()


# ──────────────────────────── Entry point ──────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on SlimeVolley-v0 then watch it play."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Total training timesteps (default: 50 000; use 0 to skip training)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Headless evaluation episodes after training (default: 5)",
    )
    parser.add_argument(
        "--vis-episodes",
        type=int,
        default=3,
        help="Rendered visualisation episodes (default: 3)",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip pygame visualisation",
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help=f"Path (without .zip) to save/load the model (default: {MODEL_DEFAULT})",
    )
    args = parser.parse_args()

    model_zip = Path(f"{args.model}.zip")

    if args.steps > 0:
        model = train(args.steps, args.model)
    elif model_zip.exists():
        print(f"\nLoading existing model from {model_zip} …")
        model = PPO.load(args.model)
    else:
        raise FileNotFoundError(
            f"--steps 0 was given but {model_zip} does not exist. "
            "Run with --steps > 0 first."
        )

    evaluate(model, n_episodes=args.eval_episodes)

    if not args.no_vis:
        visualise(model, n_episodes=args.vis_episodes)
    else:
        print("\nVisualization skipped (--no-vis).")

    print("\nDone.")


if __name__ == "__main__":
    main()
