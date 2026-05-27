"""
PPO training via AgileRL on SlimeVolley-v0, then optional visualisation.

collect_rollouts expects autoreset semantics, so we write the loop
manually to correctly handle episode boundaries with a plain gym env.

Usage:
    uv run src/train_agilerl_ppo.py                      # 10 updates × 256 steps
    uv run src/train_agilerl_ppo.py --updates 50 --learn-step 512
    uv run src/train_agilerl_ppo.py --no-vis
"""

from __future__ import annotations

import argparse

import numpy as np
from agilerl.algorithms.ppo import PPO

from slimevolleygym import BaselinePolicy, SlimeVolleyEnv

# ── Defaults (intentionally small for quick smoke-tests) ─────────────────────
LEARN_STEP = 256  # env steps per PPO update
N_UPDATES = 10  # number of updates  →  10 × 256 = 2 560 total steps
VIS_EPISODES = 3


# ── Training ─────────────────────────────────────────────────────────────────
def train(n_updates: int, learn_step: int) -> PPO:
    env = SlimeVolleyEnv()

    agent = PPO(
        observation_space=env.observation_space,
        action_space=env.action_space,
        use_rollout_buffer=True,
        num_envs=1,
        learn_step=learn_step,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        update_epochs=4,
        batch_size=64,
        device="cpu",
    )

    obs, _ = env.reset()
    current_score = 0.0
    total_steps = 0
    all_scores: list[float] = []

    for update in range(1, n_updates + 1):
        update_scores: list[float] = []
        agent.rollout_buffer.reset()
        done = False

        for _ in range(learn_step):
            # obs shape: (12,)  →  get_action expects it without a batch dim,
            # mirroring the collect_rollouts source.
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

            current_score += float(reward)
            total_steps += 1

            if done:
                update_scores.append(current_score)
                current_score = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap: value of the state we stopped at
        _, _, _, last_value = agent.get_action(obs)
        agent.rollout_buffer.compute_returns_and_advantages(
            last_value=np.atleast_1d(last_value),
            last_done=np.atleast_1d(done),
        )

        loss = agent.learn()
        all_scores.extend(update_scores)

        mean = np.mean(update_scores) if update_scores else float("nan")
        print(
            f"update {update:3d}/{n_updates}  "
            f"steps={total_steps:6d}  "
            f"ep={len(update_scores):3d}  "
            f"mean={mean:+.2f}  "
            f"loss={loss:.4f}"
        )

    env.close()
    overall = np.mean(all_scores) if all_scores else float("nan")
    print(f"\nDone — overall mean score vs baseline: {overall:+.3f}")
    return agent


# ── Visualisation ─────────────────────────────────────────────────────────────
def visualise(agent: PPO, n_episodes: int = VIS_EPISODES) -> None:
    """Render trained agent (right) vs built-in baseline (left)."""
    print(f"\nVisualising {n_episodes} episode(s) — close the window to stop.\n")
    env = SlimeVolleyEnv(render_mode="human")
    baseline = BaselinePolicy()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        obs_left = obs
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--updates",
        type=int,
        default=N_UPDATES,
        help=f"PPO updates (default {N_UPDATES})",
    )
    p.add_argument(
        "--learn-step",
        type=int,
        default=LEARN_STEP,
        help=f"Steps per update (default {LEARN_STEP})",
    )
    p.add_argument("--vis-episodes", type=int, default=VIS_EPISODES)
    p.add_argument("--no-vis", action="store_true")
    args = p.parse_args()

    total = args.updates * args.learn_step
    print(f"AgileRL PPO — SlimeVolley-v0")
    print(f"  {args.updates} updates × {args.learn_step} steps = {total:,} total\n")

    agent = train(args.updates, args.learn_step)
    if not args.no_vis:
        visualise(agent, args.vis_episodes)


if __name__ == "__main__":
    main()
