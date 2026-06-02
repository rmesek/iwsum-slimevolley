# https://docs.agilerl.com/en/latest/on_policy/index.html

from pathlib import Path

import torch
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import create_population, make_vect_envs

import slimevolleygym  # noqa: F401


def train_agent():
    # Set up paths
    base_dir = Path(__file__).parent
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    save_path = str(ckpt_dir / "ppo_slimevolley_elite.pt")
    checkpoint_base_path = str(ckpt_dir / "ppo_slimevolley")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the network configuration
    NET_CONFIG = {"head_config": {"hidden_size": [64, 64]}}

    # Define initial hyperparameters mapped to the SB3 PPO1 baseline
    POP_SIZE = 4  # Change this to >1 to seamlessly enable evolution

    INIT_HP = {
        "POP_SIZE": POP_SIZE,
        "BATCH_SIZE": 512,
        "LR": 3e-4,
        "LEARN_STEP": 512,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "ACTION_STD_INIT": 0.6,
        "CLIP_COEF": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "TARGET_KL": None,
        "UPDATE_EPOCHS": 10,
        "MAX_STEPS": 12_000_000,
        "EVO_STEPS": 10_000,
        "EVAL_STEPS": None,
        "EVAL_LOOP": 5,
        "TOURN_SIZE": 2,
        "ELITISM": True,
    }

    # Define mutation parameters
    MUT_P = {
        "NO_MUT": 1.0,
        "ARCH_MUT": 0.0,
        "NEW_LAYER": 0.0,
        "PARAMS_MUT": 0.0,
        "ACT_MUT": 0.0,
        "RL_HP_MUT": 0.0,
        "MUT_SD": 0.0,
        "RAND_SEED": 1,
    }

    if INIT_HP["POP_SIZE"] > 1:
        MUT_P["NO_MUT"] = 0.4
        MUT_P["ARCH_MUT"] = 0.2
        MUT_P["NEW_LAYER"] = 0.2
        MUT_P["PARAMS_MUT"] = 0.2
        MUT_P["ACT_MUT"] = 0.0
        MUT_P["RL_HP_MUT"] = 0.2
        MUT_P["MUT_SD"] = 0.1

    # Create the Environment
    num_envs = 8
    env_id = "SlimeVolley-v0"

    env = make_vect_envs(env_id, num_envs=num_envs)
    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # Define hyperparameter search spaces for mutations
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-3),  # type: ignore
        batch_size=RLParameter(min=64, max=512),  # type: ignore
        learn_step=RLParameter(min=256, max=1024),  # type: ignore
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
