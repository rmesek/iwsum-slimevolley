import argparse
from typing import Any, cast

import gymnasium as gym
import pygame

import shaped_slimevolleygym  # noqa: F401
import slimevolleygym  # noqa: F401


class HumanPolicy:
    def predict(self, obs):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
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


class AgileRLIPPOPolicy:
    def __init__(self, path, obs_space, act_space, agent_id="slime_1"):
        import torch
        from agilerl.algorithms import IPPO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent_id = agent_id

        self.agent = IPPO(
            observation_spaces=[obs_space, obs_space],
            action_spaces=[act_space, act_space],
            agent_ids=["slime_0", "slime_1"],
            device=self.device,
        )
        self.agent.load_checkpoint(path)

    def predict(self, obs):
        import numpy as np
        import torch

        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        dict_obs = {self.agent_id: obs}

        action_tuple = cast(Any, self.agent.get_action(dict_obs, training=False))
        action = action_tuple[0][self.agent_id]

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
        choices=[
            "baseline_vs_human",
            "elite_vs_human",
            "elite_vs_elite",
            "baseline_vs_elite",
        ],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Direct path to the .pt model file",
    )
    args = parser.parse_args()

    env = gym.make("SlimeVolley-v0", render_mode="human")
    unwrapped_env = cast(Any, env.unwrapped)
    elite_path = args.model

    policy_right: Any = None
    policy_left: Any = None

    if args.mode == "baseline_vs_human":
        policy_right = HumanPolicy()
        policy_left = unwrapped_env.policy
    elif args.mode == "elite_vs_human":
        policy_right = HumanPolicy()
        policy_left = AgileRLPolicy(elite_path)
    elif args.mode == "elite_vs_elite":
        print("Loading Self-Play Elite Agents...")
        policy_right = AgileRLIPPOPolicy(
            elite_path, env.observation_space, env.action_space, agent_id="slime_1"
        )
        policy_left = AgileRLIPPOPolicy(
            elite_path, env.observation_space, env.action_space, agent_id="slime_0"
        )
    else:
        policy_right = AgileRLPolicy(elite_path)
        policy_left = unwrapped_env.policy

    obs, info = env.reset()
    env.render()

    done = False
    while not done:
        action_right = policy_right.predict(obs)
        action_left = policy_left.predict(info["otherObs"])

        unwrapped_env.otherAction = action_left
        obs, reward, terminated, truncated, info = env.step(action_right)
        done = terminated or truncated

        env.render()
        if unwrapped_env._window is None:
            break

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
