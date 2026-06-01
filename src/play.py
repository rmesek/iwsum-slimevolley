import argparse
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import pygame

import slimevolleygym  # noqa: F401


class HumanPolicy:
    def predict(self, obs):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        # Right agent mappings: K_LEFT moves forward (towards net), K_RIGHT moves back
        return [
            int(keys[pygame.K_LEFT]),
            int(keys[pygame.K_RIGHT]),
            int(keys[pygame.K_UP]),
        ]


class AgileRLPolicy:
    def __init__(self, path):
        import torch
        from agilerl.algorithms.ppo import PPO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PPO.load(path, device=self.device)

    def predict(self, obs):
        import numpy as np
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
    pygame.display.set_caption("Slime Volleyball")

    parser = argparse.ArgumentParser(description="Slime Volleyball Player")
    parser.add_argument(
        "--mode",
        choices=["human_vs_baseline", "human_vs_elite", "elite_vs_baseline"],
        required=True,
    )
    args = parser.parse_args()

    env = gym.make("SlimeVolley-v0", render_mode="human")

    # Cast unwrapped env to Any so the type checker ignores custom attributes
    unwrapped_env = cast(Any, env.unwrapped)

    base_dir = Path(__file__).parent
    elite_path = base_dir / "checkpoints" / "ppo_slimevolley_elite.pt"

    # Pre-declare variables to satisfy the type checker
    policy_right: Any = None
    policy_left: Any = None

    if args.mode == "human_vs_baseline":
        policy_right = HumanPolicy()
        policy_left = unwrapped_env.policy
    elif args.mode == "human_vs_elite":
        policy_right = HumanPolicy()
        policy_left = AgileRLPolicy(elite_path)
    else:
        policy_right = AgileRLPolicy(elite_path)
        policy_left = unwrapped_env.policy

    obs, info = env.reset()
    env.render()

    done = False

    while not done:
        action_right = policy_right.predict(obs)
        action_left = policy_left.predict(info["otherObs"])

        # Inject the left agent's action
        unwrapped_env.otherAction = action_left

        # Step the environment wrapper with just the right agent's action
        obs, reward, terminated, truncated, info = env.step(action_right)
        done = terminated or truncated

        env.render()

        # Check if the environment destroyed the window (User clicked 'X' or pressed 'ESC')
        # If so, immediately break the loop to prevent crashes or zombie windows.
        if unwrapped_env._window is None:
            break

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
