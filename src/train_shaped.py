# https://docs.agilerl.com/en/latest/on_policy/index.html

from pathlib import Path

import torch
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import create_population, make_vect_envs

import shaped_slimevolleygym  # noqa: F401


def train_agent():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    save_path = str(ckpt_dir / "ppo_slimevolley_shaped_elite.pt")
    checkpoint_base_path = str(ckpt_dir / "ppo_slimevolley_shaped")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_envs = 16
    env_id = "SlimeVolleyShaped-v0"

    # Define the network configuration
    NET_CONFIG = {"head_config": {"hidden_size": [64, 64]}}

    # Define initial hyperparameters mapped to the SB3 PPO1 baseline
    POP_SIZE = 4  # Change this to >1 to seamlessly enable evolution

    # Define initial hyperparameters mapped to the SB3 PPO1 baseline
    INIT_HP = {
        "POP_SIZE": POP_SIZE,  # Population size (number of agents)
        "BATCH_SIZE": 256,  # Mini-batch size for network updates
        "LR": 3e-4,  # Learning rate for the optimizer
        "LEARN_STEP": 2048 // num_envs,  # Environment steps collected per iteration
        "GAMMA": 0.99,  # Reward discount factor
        "GAE_LAMBDA": 0.95,  # Generalized Advantage Estimation lambda
        "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
        "CLIP_COEF": 0.2,  # PPO policy clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient for exploration
        "VF_COEF": 0.5,  # Value function loss coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum gradient norm clipping threshold
        "TARGET_KL": 0.015,  # Target KL divergence limit
        "UPDATE_EPOCHS": 10,  # Optimization epochs per data batch
        "MAX_STEPS": 12_000_000,  # Total environment steps for training
        "EVO_STEPS": 250_000,  # Environment steps between evolutions
        "EVAL_STEPS": None,  # Evaluation episode step limit
        "EVAL_LOOP": 5,  # Number of evaluation episodes per agent
        "TOURN_SIZE": 2,  # Tournament selection pool size
        "ELITISM": True,  # Keep the best agent unchanged
    }

    # Define mutation parameters
    MUT_P = {
        "NO_MUT": 1.0,  # Probability that the agent undergoes zero mutations
        "ARCH_MUT": 0.0,  # Probability of mutating the network architecture (e.g., node count)
        "NEW_LAYER": 0.0,  # Probability of adding an entirely new hidden layer
        "PARAMS_MUT": 0.0,  # Probability of adding noise to the network's weights and biases
        "ACT_MUT": 0.0,  # Probability of changing the network's activation function
        "RL_HP_MUT": 0.0,  # Probability of mutating the RL hyperparameters (from hp_config)
        "MUT_SD": 0.0,  # Standard deviation of the Gaussian noise used for weight mutation
        "RAND_SEED": 1,  # Random seed for the mutation operations
    }

    if INIT_HP["POP_SIZE"] > 1:
        MUT_P["NO_MUT"] = 0.5
        MUT_P["ARCH_MUT"] = 0.1
        MUT_P["NEW_LAYER"] = 0.1
        MUT_P["PARAMS_MUT"] = 0.0
        MUT_P["ACT_MUT"] = 0.0
        MUT_P["RL_HP_MUT"] = 0.3
        MUT_P["MUT_SD"] = 0.1

    # Create the Environment using the custom shaped ID
    env = make_vect_envs(env_id, num_envs=num_envs)
    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # Define hyperparameter search spaces for mutations
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-5, max=1e-3),  # type: ignore
        batch_size=RLParameter(min=64, max=512, dtype=int),  # type: ignore
        learn_step=RLParameter(min=512, max=4096, dtype=int),  # type: ignore
        ent_coef=RLParameter(min=0.0, max=0.05),  # type: ignore
    )

    # Create a Population of Agents
    pop = create_population(
        algo="PPO",
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        hp_config=hp_config,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Create Tournament and Mutation Objects
    tournament = TournamentSelection(
        tournament_size=INIT_HP["TOURN_SIZE"],
        elitism=INIT_HP["ELITISM"],
        population_size=INIT_HP["POP_SIZE"],
        eval_loop=INIT_HP["EVAL_LOOP"],
    )

    mutations = Mutations(
        no_mutation=MUT_P["NO_MUT"],
        architecture=MUT_P["ARCH_MUT"],
        new_layer_prob=MUT_P["NEW_LAYER"],
        parameters=MUT_P["PARAMS_MUT"],
        activation=MUT_P["ACT_MUT"],
        rl_hp=MUT_P["RL_HP_MUT"],
        mutation_sd=MUT_P["MUT_SD"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )

    print(f"Starting AgileRL Training on {env_id}...")

    # Training and Saving an Agent
    trained_pop, pop_fitnesses = train_on_policy(
        env=env,
        env_name=env_id,
        algo="PPO",
        pop=pop,  # type: ignore
        max_steps=INIT_HP["MAX_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        evo_steps=INIT_HP["EVO_STEPS"],
        tournament=tournament,
        mutation=mutations,
        wb=False,
        # Built-in checkpointing args:
        checkpoint=100_000,
        checkpoint_path=checkpoint_base_path,
        save_elite=True,
        elite_path=save_path,
    )

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    train_agent()
