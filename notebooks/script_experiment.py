import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Union
import random
from types import SimpleNamespace

from rl_final_project.control import AbstractControl
from rl_final_project.q_functions import QLinear, QTabular
from rl_final_project.control import MonteCarloControl, QLearningControl, \
    SarsaLambdaControl
from rl_final_project.agent import Agent
from rl_final_project.dqn import DQNFunction, DQNControl
import pickle

import multiprocessing

n_cpu = multiprocessing.cpu_count()

from copy import copy

cmap = {
    "sarsa-lambda": SarsaLambdaControl,
    "q-learning": QLearningControl,
    "dqn": DQNControl,
    "monte-carlo": MonteCarloControl,
}

fmap = {
    "linear": QLinear,
    "tabular": QTabular,
}


def build_control_experiment(
        env: gym.Env,
        method: str,
        gamma: float,
        reward_mode: str,
        function: Optional[str] = None,
        num_episodes: int = 10_000,
        replay_capacity: int = 10_000,
        n0: int = 10,
        discrete_scale: int = 1,
        batch_size: int = 128,
        eps_func: str = "dqn",
        stochasticity_factor: float = 0.4,
        method_args: Optional[dict] = None,
) -> AbstractControl:
    if method_args is None:
        method_args = {}

    if function is None and method != "dqn":
        raise ValueError(
            "function must be specified for all methods except dqn")

    if function is not None and function not in fmap:
        raise ValueError(f"Unknown function approximation {function}")

    if method not in cmap:
        raise ValueError(f"Unknown control method {method}")

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    if method == "dqn":
        q_function = DQNFunction(
            batch_size=batch_size,
            n_actions=n_actions,
            n_feat=n_states,
            discrete_scale=discrete_scale,
        )
    else:
        q_function = fmap[function](
            n_actions=n_actions,
            n_feat=n_states,
            discrete_scale=discrete_scale
        )

    agent = Agent(
        q_function,
        n0=n0,
        n_actions=n_actions,
        eps_greedy_function=eps_func,
        stochasticity_factor=stochasticity_factor,
    )

    for k in copy(method_args):
        if method_args[k] is None:
            del method_args[k]

    control = cmap[method](
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        gamma=gamma,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        verbose=False,
        reward_mode=reward_mode,
        **method_args
    )

    return control

from rl_final_project.environment import EnvironmentNormalizer


class ExperimentExitCode:
    SUCCESS = 0
    FAILED = 1
    INVALID = 2


def run(config: SimpleNamespace) -> tuple[
    ExperimentExitCode, SimpleNamespace, Union[list[float], Exception]]:
    # Initialize random seed
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Initialize environment
    env = gym.make(config.env_name)
    env = EnvironmentNormalizer(env)

    try:
        # Initialize control algorithm
        control_algorithm = build_control_experiment(
            env=env,
            method=config.control_algorithm.method,
            gamma=config.gamma,
            reward_mode=config.reward_mode,
            function=config.control_algorithm.function,
            num_episodes=config.num_episodes,
            replay_capacity=config.replay_capacity,
            batch_size=config.batch_size,
            n0=config.n0,
            discrete_scale=config.discrete_scale,
            eps_func=config.eps_func,
            stochasticity_factor=config.stochasticity_factor,
            method_args=config.control_algorithm.args.__dict__,
        )
    except Exception as e:
        return ExperimentExitCode.INVALID, config, e, 0

    # Run control algorithm
    try:
        eps_rewards = control_algorithm.fit()
    except Exception as e:
        return ExperimentExitCode.FAILED, config, e, 0

    try:
        n_visited = control_algorithm.agent.q_function.states_explored
    except Exception:
        n_visited = 0

    return ExperimentExitCode.SUCCESS, config, eps_rewards, n_visited

def experiment_generator_grid_search() -> SimpleNamespace:
    for env_name in ["CartPole-v1"]:
        for method in ["q-learning", "sarsa-lambda", "monte-carlo", "dqn"]:
            functions = ["linear", "tabular"] if method != "dqn" else [None]
            lbds = [0.0, 0.3, 0.6, 0.9] if method == "sarsa-lambda" else [None]
            n0s = [2, 5, 10, 25] if method != "dqn" else [None]
            eps_func = "s" if method != "dqn" else "dqn"
            scales = [2, 5, 20] if method != "dqn" else [1]
            stochasticity_factors = [0]
            lrs = [0.0001] if method == "dqn" else [None]
            taus = [0.005] if method == "dqn" else [None]
            modes = ["sparse", "default"] if method == "dqn" else ["sparse"]
            for function in functions:
                for gamma in [0.5, 0.95]:
                    for lbd in lbds:
                        for n0 in n0s:
                            for discrete_scale in scales:
                                for stochasticity_factor in stochasticity_factors:
                                    for reward_mode in modes:
                                        for lr in lrs:
                                            for tau in taus:
                                                for seed in range(3):
                                                    if True:
                                                        yield SimpleNamespace(
                                                            env_name=env_name,
                                                            num_episodes=10_000 if method != "dqn" else 800,
                                                            control_algorithm=SimpleNamespace(
                                                                method=method,
                                                                args=SimpleNamespace(
                                                                    lambda_factor=lbd,
                                                                    lr=lr,
                                                                    tau=tau,
                                                                ),
                                                                function=function,
                                                            ),
                                                            gamma=gamma,
                                                            replay_capacity=10_000,
                                                            batch_size=None if method != "dqn" else 128,
                                                            discrete_scale=discrete_scale,
                                                            n0=n0,
                                                            eps_func=eps_func,
                                                            stochasticity_factor=stochasticity_factor,
                                                            reward_mode=reward_mode,
                                                            seed=seed
                                                        )


from multiprocessing import Pool

experiments = list(experiment_generator_grid_search())
results = []

with Pool(n_cpu) as p:
    for exit_code, config, eps_rewards, n_visited in tqdm(
            p.imap_unordered(run, experiments), total=len(experiments)):
        results.append((exit_code, config, eps_rewards, n_visited))
        with open("results_sparse.pkl", "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

