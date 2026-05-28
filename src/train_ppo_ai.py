#!/usr/bin/env python3

import math

import gymnasium as gym
import numpy as np
import torch
from agilerl.algorithms.ppo import PPO
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

import slimevolleygym
from slimevolleygym import BaselinePolicy

# --- Constants & Hyperparameters ---
NUM_TIMESTEPS = int(2e7)
SEED = 721
NUM_ENVS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Utilities ---
def to_binary(act):
    # Safely extract the pure integer from arrays, lists, or tensors
    act_val = int(np.asarray(act).item())
    return np.array([int(x) for x in format(act_val, "03b")], dtype=np.int8)


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, act):
        return to_binary(act)


# --- 1. Setup Training Environment ---
env = SyncVectorEnv(
    [lambda: DiscreteActionWrapper(gym.make("SlimeVolley-v0")) for _ in range(NUM_ENVS)]
)
env.action_space.seed(SEED)

# --- 2. Initialize AgileRL PPO Agent ---
agent = PPO(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    device=DEVICE,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.0,
    batch_size=128,
    learn_step=2048,
)

# --- 3. Custom Training Loop ---
print(f"Starting custom training loop for {NUM_TIMESTEPS} timesteps...")
pbar = tqdm(total=NUM_TIMESTEPS)

steps_per_env = math.ceil(agent.learn_step / NUM_ENVS)
steps_taken = 0

obs, _ = env.reset(seed=SEED)
done = np.zeros(NUM_ENVS)

while steps_taken < NUM_TIMESTEPS:
    observations, actions, log_probs = [], [], []
    rewards, dones, values = [], [], []

    agent.set_training_mode(True)

    for _ in range(steps_per_env):
        action, log_prob, _, value = agent.get_action(obs)

        next_obs, reward, term, trunc, _ = env.step(action)
        next_done = np.logical_or(term, trunc).astype(np.int8)

        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        obs, done = next_obs, next_done

    agent.learn(
        (observations, actions, log_probs, rewards, dones, values, next_obs, next_done)
    )

    # Update progress
    batch_steps = steps_per_env * NUM_ENVS
    steps_taken += batch_steps
    pbar.update(batch_steps)
    pbar.set_description(f"Batch Mean Reward: {np.mean(np.sum(rewards, axis=0)):.2f}")

pbar.close()
env.close()

# --- 4. Infinite Evaluation Loop ---
print("\nTraining complete! Starting infinite bot-play visualization...")

eval_env = gym.make("SlimeVolley-v0", render_mode="human")
obs_right, _ = eval_env.reset(seed=SEED)
obs_left = obs_right

policy_left = BaselinePolicy()

while True:
    # Right Agent (AgileRL)
    action_out = agent.get_action(obs_right)
    action_discrete = action_out[0] if isinstance(action_out, tuple) else action_out
    action_right = to_binary(action_discrete)

    # Left Agent (Baseline)
    action_left = policy_left.predict(obs_left)

    # Step bypassing Gym's strict signature
    obs_right, _, terminated, truncated, info = eval_env.unwrapped.step(
        action_right, action_left
    )  # type: ignore
    obs_left = info["otherObs"]

    eval_env.render()

    if terminated or truncated:
        obs_right, _ = eval_env.reset()
        obs_left = obs_right

eval_env.close()
