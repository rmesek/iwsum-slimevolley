"""
PPO training via AgileRL on SlimeVolley-v0 (PPO agent = right, baseline = left).
Hyperparameters adapted from the original Stable Baselines PPO1 example.

Usage:
    uv run src/train_agilerl_ppo.py                  # defaults to 20M steps
    uv run src/train_agilerl_ppo.py --steps 3000000  # 3M steps (should beat baseline)
    uv run src/train_agilerl_ppo.py --no-vis         # headless only

Training time guide (single CPU):
    It takes roughly 3,000,000 steps to solve the environment (beat the built-in AI).
"""

from __future__ import annotations

import argparse

import numpy as np
from agilerl.algorithms.ppo import PPO
from tqdm import tqdm

from slimevolleygym import BaselinePolicy, SlimeVolleyEnv

LEARN_STEP = 4096  # steps per PPO update
BATCH_SIZE = 64
LR = 3e-4


# ── Training ──────────────────────────────────────────────────────────────────
def train(max_steps: int) -> PPO:
    env = SlimeVolleyEnv()

    agent = PPO(
        observation_space=env.observation_space,
        action_space=env.action_space,
        use_rollout_buffer=True,
        num_envs=1,
        learn_step=LEARN_STEP,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        update_epochs=10,
        device="cpu",
    )

    # Carry env state across update boundaries — correct GAE bootstrapping.
    obs, _ = env.reset()
    done = False
    ep_score = 0.0
    all_scores: list[float] = []
    total_steps = 0
    pbar = tqdm(total=max_steps, unit="step")

    while total_steps < max_steps:
        agent.rollout_buffer.reset()

        for _ in range(LEARN_STEP):
            action, log_prob, _, value = agent.get_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action.squeeze(0))
            done = bool(terminated or truncated)

            agent.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=np.atleast_1d(float(reward)),
                done=np.atleast_1d(done),
                value=np.atleast_1d(value),
                log_prob=np.atleast_1d(log_prob),
                next_obs=next_obs,
            )

            ep_score += float(reward)

            if done:
                all_scores.append(ep_score)
                ep_score = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        _, _, _, last_value = agent.get_action(obs)
        agent.rollout_buffer.compute_returns_and_advantages(
            last_value=np.atleast_1d(last_value),
            last_done=np.atleast_1d(done),
        )

        loss = agent.learn()

        total_steps += LEARN_STEP
        pbar.update(LEARN_STEP)
        recent = np.mean(all_scores[-20:]) if all_scores else float("nan")
        pbar.set_postfix(loss=f"{loss:.4f}", score=f"{recent:+.2f}")

    pbar.close()
    env.close()

    overall = np.mean(all_scores) if all_scores else float("nan")
    print(f"\nDone — mean score vs baseline: {overall:+.3f}")
    return agent


# ── Visualisation ─────────────────────────────────────────────────────────────
def visualise(agent: PPO, n_episodes: int = 3) -> None:
    """Trained PPO (right) vs built-in baseline (left)."""
    print(f"\nPPO (right) vs Baseline (left) — {n_episodes} episode(s)\n")
    env = SlimeVolleyEnv(render_mode="human")
    baseline = BaselinePolicy()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=100 + ep)
        obs_left = info["otherObs"]
        baseline.reset()
        done, score = False, 0.0

        while not done:
            action, _, _, _ = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0), baseline.predict(obs_left)
            )
            obs_left = info["otherObs"]
            score += float(reward)
            done = bool(terminated or truncated)
            env.render()

        result = "win" if score > 0 else ("loss" if score < 0 else "draw")
        print(f"  episode {ep + 1}: {score:+.0f}  [{result}]")

    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="PPO (AgileRL) on SlimeVolley-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--steps", type=int, default=20_000_000, help="Total training env steps"
    )
    p.add_argument("--vis-episodes", type=int, default=3)
    p.add_argument("--no-vis", action="store_true")
    args = p.parse_args()

    print(f"AgileRL PPO — SlimeVolley-v0")
    print(f"  learn_step={LEARN_STEP}   max_steps={args.steps:,}\n")

    agent = train(args.steps)
    if not args.no_vis:
        visualise(agent, args.vis_episodes)


if __name__ == "__main__":
    main()
