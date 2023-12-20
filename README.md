# Basic Reinforcement Learning Framework.

This is a project developed and evaluated in the course of Reinforcement Learning at the State University of Campinas.

[![Tests](https://github.com/ilan-francisco/basic-reinforcement-learning-framework/workflows/Tests/badge.svg)][tests]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[tests]: https://github.com/ilan-francisco/basic-reinforcement-learning-framework/actions?workflow=Tests
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- Furama's Gymnasium integration
- Monte-carlo Classic Control
- Q-Learning Classic Control
- Sarsa-lambda Classic Control
- DQN

## Requirements

This was tested with the following packages:

- torch==2.0.1
- gymnasium==0.29.1
- scikit-learn==1.3.2
- tqdm==4.66.1
- numba==0.58.1
- matplotlib==3.8.2

## Installation

You can install it by the following command (I did not have time to test it):

```console
$ pip3 install https://github.com/ilan-francisco/basic-reinforcement-learning-framework.git
```

## Usage

Here is an example of basic usage:

```python
import matplotlib.pyplot as plt

from rl_final_project.agent import Agent
from rl_final_project.dqn import DQNControl
from rl_final_project.dqn import DQNFunction
from rl_final_project.environment import EnvironmentNormalizer

env = EnvironmentNormalizer.from_gym("CartPole-v1")
n_states = env.observation_space.shape[0]

q_function = DQNFunction(
    n_actions=env.action_space.n, 
    n_feat=n_states,
    batch_size=128
)

agent = Agent(
    q_function,
    n_actions=env.action_space.n,
    eps_greedy_function="dqn",
    stochasticity_factor=0.0,
)

control = DQNControl(
    lr=0.001, 
    tau=0.005, 
    env=env, 
    agent=agent,
    num_episodes=1000, 
    gamma=0.99, 
    batch_size=128, 
    reward_mode="default"
)

eps_rewards = control.fit()

# plot the results
import matplotlib.pyplot as plt
plt.plot(eps_rewards)
plt.title("Monte-Carlo Control with DQN")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Basic Reinforcement Learning Framework._ is free and open-source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

<!-- github-only -->

[license]: https://github.com/ilan-francisco/mo436_final_project/blob/main/LICENSE
[contributor guide]: https://github.com/ilan-francisco/mo436_final_project/blob/main/CONTRIBUTING.md
[command-line reference]: https://mo436_final_project.readthedocs.io/en/latest/usage.html
