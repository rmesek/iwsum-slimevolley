from pathlib import Path
from typing import Any, Dict, cast

import gymnasium as gym
import numpy as np
import torch
from agilerl.algorithms import IPPO
from tqdm import tqdm

import shaped_slimevolleygym  # noqa: F401


def evaluate_against_baseline(agent, env_id="SlimeVolleyShaped-v0", num_episodes=10):
    """
    Evaluates the IPPO agent's absolute performance against the built-in baseline policy.
    Self-play scores are relative; this provides a true metric of skill progression.
    """
    eval_env = gym.make(env_id)
    returns = []
    for _ in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_return = 0
        while not done:
            # Baseline policy controls the left agent
            eval_env.unwrapped.otherAction = eval_env.unwrapped.policy.predict(
                info["otherObs"]
            )

            # IPPO agent controls the right agent
            obs_expanded = np.expand_dims(obs, axis=0)
            dict_obs = {"slime_1": obs_expanded}

            # Safely unpack action dictionary whether it's wrapped in a tuple or not
            action_out = agent.get_action(dict_obs, training=False)
            dict_actions = (
                action_out[0] if isinstance(action_out, tuple) else action_out
            )
            action = dict_actions["slime_1"][0]

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            obs, reward, term, trunc, info = eval_env.step(action)
            episode_return += reward
            done = term or trunc

        returns.append(episode_return)

    eval_env.close()
    return np.mean(returns)


def train_multiagent_ippo():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(ckpt_dir / "ippo_slimevolley_shaped_elite.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 8

    # Create a synchronous list of environments
    envs = [gym.make("SlimeVolleyShaped-v0") for _ in range(num_envs)]

    single_obs_space = envs[0].observation_space
    single_act_space = envs[0].action_space

    # IPPO Setup for 2 Homogeneous Agents
    agent_ids = ["slime_0", "slime_1"]
    observation_spaces = [single_obs_space, single_obs_space]
    action_spaces = [single_act_space, single_act_space]

    agent = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
        batch_size=256,
        lr=3e-4,
        learn_step=512,
        gamma=0.99,
        ent_coef=0.01,
    )

    t_agent = cast(Any, agent)
    max_steps = 20_000_000
    checkpoint_freq = 250_000

    pbar = tqdm(total=max_steps)

    obs_dim = single_obs_space.shape[0]
    obs_right = np.zeros((num_envs, obs_dim), dtype=np.float32)
    obs_left = np.zeros((num_envs, obs_dim), dtype=np.float32)

    for i, env in enumerate(envs):
        o_r, info = env.reset()
        obs_right[i] = o_r
        obs_left[i] = info["otherObs"]

    scores = np.zeros(num_envs)
    completed_scores = []
    last_checkpoint = 0
    best_eval_score = -np.inf  # Track best score for elite saving

    print("Starting IPPO Self-Play Training...")

    while t_agent.steps[-1] < max_steps:
        states = {a: [] for a in agent_ids}
        actions = {a: [] for a in agent_ids}
        log_probs = {a: [] for a in agent_ids}
        entropies = {a: [] for a in agent_ids}
        rewards = {a: [] for a in agent_ids}
        values = {a: [] for a in agent_ids}
        dones = {a: [] for a in agent_ids}

        steps_added = 0

        for _ in range(t_agent.learn_step):
            dict_obs = {
                "slime_0": obs_left,
                "slime_1": obs_right,
            }

            action_tuple = agent.get_action(obs=dict_obs, training=True)

            dict_actions = cast(Dict[str, np.ndarray], action_tuple[0])
            dict_log_probs = cast(Dict[str, np.ndarray], action_tuple[1])
            dict_entropies = cast(Dict[str, np.ndarray], action_tuple[2])
            dict_values = cast(Dict[str, np.ndarray], action_tuple[3])

            next_obs_right = np.zeros((num_envs, obs_dim), dtype=np.float32)
            next_obs_left = np.zeros((num_envs, obs_dim), dtype=np.float32)
            step_rewards_right = np.zeros(num_envs, dtype=np.float32)
            step_terms = np.zeros(num_envs, dtype=bool)

            for i, env in enumerate(envs):
                env.unwrapped.otherAction = dict_actions["slime_0"][i]
                o_r, r_r, term, trunc, info = env.step(dict_actions["slime_1"][i])

                step_rewards_right[i] = r_r
                step_terms[i] = term or trunc
                scores[i] += r_r

                if term or trunc:
                    completed_scores.append(scores[i])
                    scores[i] = 0
                    o_r, info = env.reset()

                next_obs_right[i] = o_r
                next_obs_left[i] = info["otherObs"]

            # Strict zero-sum enforcement for symmetric self-play
            step_rewards_left = -step_rewards_right

            dict_rewards = {"slime_0": step_rewards_left, "slime_1": step_rewards_right}
            dict_dones = {"slime_0": step_terms, "slime_1": step_terms}

            for a_id in agent_ids:
                states[a_id].append(dict_obs[a_id])
                actions[a_id].append(dict_actions[a_id])
                log_probs[a_id].append(dict_log_probs[a_id])
                entropies[a_id].append(dict_entropies[a_id])
                values[a_id].append(dict_values[a_id])
                rewards[a_id].append(dict_rewards[a_id])
                dones[a_id].append(dict_dones[a_id])

            obs_right = next_obs_right
            obs_left = next_obs_left
            steps_added += num_envs

        dict_next_obs = {"slime_0": obs_left, "slime_1": obs_right}
        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            dict_next_obs,
            dict_dones,
        )
        t_agent.learn(experiences)
        current_steps = t_agent.steps[-1] + steps_added
        t_agent.steps[-1] = current_steps

        pbar.update(steps_added)

        # Periodic Checkpointing & Baseline Evaluation
        if current_steps - last_checkpoint >= checkpoint_freq:
            eval_score = evaluate_against_baseline(
                agent, env_id="SlimeVolleyShaped-v0", num_episodes=10
            )

            pbar.set_description(
                f"Eval vs Baseline: {eval_score:.2f} | Self-Play: {np.mean(completed_scores[-20:]):.2f}"
            )

            ckpt_path = str(ckpt_dir / f"ippo_checkpoint_{current_steps}.pt")
            agent.save_checkpoint(ckpt_path)

            # Only overwrite the elite model if it actually improved
            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                agent.save_checkpoint(save_path)
                tqdm.write(
                    f"\n[Step {current_steps}] New elite model saved! Eval Score: {eval_score:.2f}"
                )

            last_checkpoint = current_steps

    for env in envs:
        env.close()
    print("Training complete!")


if __name__ == "__main__":
    train_multiagent_ippo()
