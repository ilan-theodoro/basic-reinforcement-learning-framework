"""Test cases for the environment module."""
import numpy as np
import pytest

from rl_final_project.environment import EnvironmentNormalizer


@pytest.mark.usefixtures("available_envs")
def test_environment_normalization(available_envs: list) -> None:
    """Test environment normalization."""
    for unnorm_env in available_envs:
        env = EnvironmentNormalizer(unnorm_env)
        right_edge = env._normalize(unnorm_env.observation_space.high)
        left_edge = env._normalize(unnorm_env.observation_space.low)
        unit_vector = np.ones_like(right_edge)
        assert np.all(
            np.isclose(right_edge, unit_vector)
        ), f"Expected unit vector, got {right_edge}"
        assert np.all(
            np.isclose(left_edge, -unit_vector)
        ), f"Expected negative unit vector, got {left_edge}"

        obs, _ = env.reset()
        assert np.all(obs >= -1)
        assert np.all(obs <= 1)
        env.close()
