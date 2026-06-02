import argparse
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pygame

import shaped_slimevolleygym  # noqa: F401
import slimevolleygym  # noqa: F401


class HumanPolicy:
    def predict(self, obs: np.ndarray) -> list[int]:
        keys = pygame.key.get_pressed()
        return [
            int(keys[pygame.K_LEFT]),
            int(keys[pygame.K_RIGHT]),
            int(keys[pygame.K_UP]),
        ]


class AgileRLPolicy:
    def __init__(self, path: str):
        import torch
        from agilerl.algorithms.ppo import PPO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PPO.load(path, device=self.device)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        import torch

        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        action, *rest = self.model.get_action(obs, training=False)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        return action[0]


def main():
    pygame.init()
    pygame.display.set_mode((1200, 500))
    pygame.display.set_caption("Slime Volleyball Player")

    parser = argparse.ArgumentParser(description="Slime Volleyball Match Viewer")
    parser.add_argument(
        "--mode",
        choices=["baseline_vs_human", "elite_vs_human", "baseline_vs_elite"],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Direct path to the .pt model file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="SlimeVolley-v0",
        help="The Gym environment ID to play in.",
    )
    args = parser.parse_args()

    # UX Improvement: Auto-switch environment if loading a shaped model
    if "shaped" in args.model.lower() and args.env == "SlimeVolley-v0":
        print("Auto-detecting shaped environment based on model name...")
        args.env = "SlimeVolleyShaped-v0"

    env = gym.make(args.env, render_mode="human")
    unwrapped_env = cast(Any, env.unwrapped)
    # unwrapped_env.t_limit = float("inf")

    if args.mode == "baseline_vs_human":
        policy_right = HumanPolicy()
        policy_left = unwrapped_env.policy
    elif args.mode == "elite_vs_human":
        policy_right = HumanPolicy()
        policy_left = AgileRLPolicy(args.model)
    else:
        policy_right = AgileRLPolicy(args.model)
        policy_left = unwrapped_env.policy

    obs, info = env.reset()
    env.render()

    done = False
    while not done:
        # Proper OS-level event loop handling avoids hanging/freezing windows
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if done:
            break

        action_right = policy_right.predict(obs)
        action_left = policy_left.predict(info["otherObs"])

        unwrapped_env.otherAction = action_left
        obs, reward, terminated, truncated, info = env.step(action_right)
        done = terminated or truncated

        env.render()

        # Original fallback condition if Pygame window is directly destroyed
        if unwrapped_env._window is None:
            break

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
