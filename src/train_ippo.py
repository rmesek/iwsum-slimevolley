"""
IPPO self-play on SlimeVolley-v0 (AgileRL).

Both agents are trained simultaneously with Independent PPO against each
other (self-play).  The env normalises all observations to the right-agent
perspective, so the same policy generalises to both sides.

Usage:
    uv run src/train_ippo.py                     # 10 000 steps, then watch bots play
    uv run src/train_ippo.py --steps 500000      # ~9 min on CPU, first real learning
    uv run src/train_ippo.py --steps 500000 --human   # play against the bot afterwards
    uv run src/train_ippo.py --no-vis            # headless only

Training time guide (single CPU):
    10 000 steps  →  ~10 s   (smoke test)
   200 000 steps  →  ~3 min  (first signs of learning)
   500 000 steps  →  ~9 min  (competent self-play)
 2 000 000 steps  →  ~35 min (strong agent)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from agilerl.algorithms import IPPO
from tqdm import tqdm

from slimevolleygym import SlimeVolleyEnv

# ── Constants ─────────────────────────────────────────────────────────────────
AGENT_IDS = ["right", "left"]
LEARN_STEP = 512  # env steps between updates — raise for better learning
BATCH_SIZE = 64
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Training ──────────────────────────────────────────────────────────────────
def train(max_steps: int) -> IPPO:
    env = SlimeVolleyEnv()

    agent = IPPO(
        observation_spaces=[env.observation_space] * 2,
        action_spaces=[env.action_space] * 2,
        agent_ids=AGENT_IDS,
        learn_step=LEARN_STEP,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        update_epochs=4,
        device=str(device),
    )

    # Carry env state across update boundaries — correct GAE bootstrapping.
    obs_r, info = env.reset()
    obs = {"right": obs_r, "left": info["otherObs"]}
    done = {aid: np.zeros(1) for aid in AGENT_IDS}

    ep_score = 0.0
    completed_scores: list[float] = []
    total_steps = 0
    pbar = tqdm(total=max_steps, unit="step")

    while total_steps < max_steps:
        states = {aid: [] for aid in AGENT_IDS}
        actions = {aid: [] for aid in AGENT_IDS}
        log_probs = {aid: [] for aid in AGENT_IDS}
        rewards = {aid: [] for aid in AGENT_IDS}
        dones = {aid: [] for aid in AGENT_IDS}
        values = {aid: [] for aid in AGENT_IDS}

        for _ in range(LEARN_STEP):
            act, lp, _, val = agent.get_action(obs=obs)

            next_r, reward, terminated, truncated, step_info = env.step(
                act["right"].squeeze(), act["left"].squeeze()
            )
            next_l = step_info["otherObs"]
            is_done = bool(terminated or truncated)

            for aid, r in zip(AGENT_IDS, [reward, -reward]):
                states[aid].append(obs[aid])
                actions[aid].append(act[aid].squeeze())
                log_probs[aid].append(lp[aid].squeeze())
                rewards[aid].append(np.array([r]))
                dones[aid].append(done[aid])  # done from PREVIOUS step
                values[aid].append(val[aid].squeeze())

            ep_score += reward

            if is_done:
                completed_scores.append(ep_score)
                ep_score = 0.0
                next_r, reset_info = env.reset()
                next_l = reset_info["otherObs"]

            obs = {"right": next_r, "left": next_l}
            done = {aid: np.array([float(is_done)]) for aid in AGENT_IDS}

        loss = agent.learn(
            (states, actions, log_probs, rewards, dones, values, obs, done)
        )

        total_steps += LEARN_STEP
        pbar.update(LEARN_STEP)
        recent = np.mean(completed_scores[-20:]) if completed_scores else float("nan")
        mean_loss = float(
            np.mean(
                [v.item() if hasattr(v, "item") else float(v) for v in loss.values()]
            )
        )
        pbar.set_postfix(loss=f"{mean_loss:.4f}", score=f"{recent:+.2f}")

    pbar.close()
    env.close()
    return agent


# ── Bot vs Bot visualisation ──────────────────────────────────────────────────
def visualise(agent: IPPO, n_episodes: int = 3) -> None:
    print(f"\nIPPO right vs IPPO left — {n_episodes} episode(s)\n")
    env = SlimeVolleyEnv(render_mode="human")

    for ep in range(n_episodes):
        obs_r, info = env.reset(seed=ep)
        obs = {"right": obs_r, "left": info["otherObs"]}
        done, score = False, 0.0

        while not done:
            act, _, _, _ = agent.get_action(obs=obs)
            obs_r, reward, terminated, truncated, info = env.step(
                act["right"].squeeze(), act["left"].squeeze()
            )
            obs = {"right": obs_r, "left": info["otherObs"]}
            score += float(reward)
            done = bool(terminated or truncated)
            env.render()

        print(f"  episode {ep + 1}: {score:+.0f}")

    env.close()


# ── Human vs trained bot ──────────────────────────────────────────────────────
def play_human(agent: IPPO) -> None:
    """Human (left slime, keyboard) vs trained IPPO (right slime).

    Controls
    --------
    A / ← arrow  move left (away from fence)
    D / → arrow  move right (toward fence)
    W / ↑ arrow  jump
    Q / Esc      quit
    """
    import pygame  # only needed when rendering

    env = SlimeVolleyEnv(render_mode="human")
    obs_r, info = env.reset()
    obs = {"right": obs_r, "left": info["otherObs"]}
    sr = sl = 0.0

    print("\nHuman (left, blue)  vs  IPPO (right, yellow)")
    print("  A/← move left   D/→ move right   W/↑ jump   Q/Esc quit\n")

    try:
        while True:
            env.render()
            if env._window is None:  # window was closed via ×
                break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                break

            # Left slime: action = [forward=toward_fence, backward, jump]
            human = [
                int(keys[pygame.K_d] or keys[pygame.K_RIGHT]),  # forward (right)
                int(keys[pygame.K_a] or keys[pygame.K_LEFT]),  # backward (left)
                int(keys[pygame.K_w] or keys[pygame.K_UP]),  # jump
            ]

            act, _, _, _ = agent.get_action(obs=obs)
            obs_r, reward, terminated, truncated, info = env.step(
                act["right"].squeeze(), human
            )
            obs = {"right": obs_r, "left": info["otherObs"]}
            sr += float(reward)
            sl -= float(reward)  # zero-sum

            if terminated or truncated:
                print(f"  IPPO: {sr:+.0f}   Human: {sl:+.0f}")
                obs_r, info = env.reset()
                obs = {"right": obs_r, "left": info["otherObs"]}
                sr = sl = 0.0
    finally:
        env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="IPPO self-play on SlimeVolley-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=10_000, help="Total training env steps")
    p.add_argument("--vis-episodes", type=int, default=3)
    p.add_argument(
        "--no-vis", action="store_true", help="Skip bot-vs-bot visualisation"
    )
    p.add_argument(
        "--human",
        action="store_true",
        help="Play against the trained bot after training",
    )
    args = p.parse_args()

    print(f"IPPO self-play — SlimeVolley-v0   device={device}")
    print(f"  learn_step={LEARN_STEP}   max_steps={args.steps:,}\n")

    agent = train(args.steps)

    if not args.no_vis:
        visualise(agent, args.vis_episodes)

    if args.human:
        play_human(agent)


if __name__ == "__main__":
    main()
