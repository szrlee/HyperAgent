'''
From DeepMind: https://github.com/deepmind/bsuite/blob/afdeae850b08108d2247a1802567bb7f404d9833/bsuite/environments/deep_sea.py
'''

import gym
import numpy as np
import warnings
from typing import Optional
from gym.spaces import Discrete, Box


class DeepSea(gym.Env):
    """Deep Sea environment to test for deep exploration."""

    def __init__(
        self,
        size: int = 10,
        # deterministic: bool = True,
        random_reward_scale: float = 0.,
        random_dynamics_scale: float = 0.,
        unscaled_move_cost: float = 0.01,
        other_reward: float = 0.0,
        randomize_actions: bool = True,
        seed: Optional[int] = None,
        mapping_seed: Optional[int] = None
    ):
        """Deep sea environment to test for deep exploration.
        Args:
            size: The size of `N` for the N x N grid of states.
            deterministic: Whether transitions are deterministic (default) or 'windy',
            i.e. the `right` action fails with probability 1/N.
            unscaled_move_cost: The move cost for moving right, multiplied by N. The
            default (0.01) means the optimal policy gets 0.99 episode return.
            randomize_actions: The definition of DeepSea environment includes random
            mappings of actions: (0,1) -> (left, right) by state. For debugging
            purposes, we include the option to turn this randomization off and
            let 0=left, 1=right in every state.
            seed: Random seed for rewards and transitions, if applicable.
            mapping_seed: Random seed for action mapping, if applicable.
        """
        super().__init__()
        self._size = size
        # self._deterministic = deterministic
        self._random_reward_scale = random_reward_scale
        self._random_dynamics_scale = random_dynamics_scale
        self._unscaled_move_cost = unscaled_move_cost
        self._other_reward = other_reward
        self._rng = np.random.RandomState(seed)

        if randomize_actions:
            self._mapping_rng = np.random.RandomState(mapping_seed)
            self._action_mapping = self._mapping_rng.binomial(1, 0.5, [size, size])
        else:
            warnings.warn('Environment is in debug mode (randomize_actions=False).'
                        'Only randomized_actions=True is the DeepSea environment.')
            self._action_mapping = np.ones([size, size])

        self._random_dynamics = (1 / size) * random_dynamics_scale
        self._deterministic = True if random_dynamics_scale == 0 else False
        if not self._deterministic:  # action 'right' only succeeds (1 - 1/N)
            optimal_no_cost = (1 - 1 / self._size) ** (self._size - 1)
        else:
            optimal_no_cost = 1.
        self._optimal_return = optimal_no_cost - self._unscaled_move_cost

        self._row = 0
        self._column = 0

        self.observation_space = Box(0, 1, shape=(size, size), dtype=np.uint8)
        self.action_space = Discrete(n=2)

    def _get_action_mapping(self):
        return self._action_mapping.flatten()

    def _get_observation(self):
        obs = np.zeros(shape=(self._size, self._size), dtype=np.uint8)
        if self._row >= self._size:  # End of episode null observation
            return obs # .flatten()
        obs[self._row, self._column] = 1
        return obs # .flatten()

    def reset(self):
        self._row = 0
        self._column = 0
        return self._get_observation()

    def step(self, action: int):
        reward = 0.
        action_right = action == self._action_mapping[self._row, self._column]

        # Reward calculation
        if self._row == self._size - 1:
            if (self._column == self._size -1 and not action_right) or \
                (self._column != self._size - 1):
                reward += self._other_reward
        if self._column == self._size - 1 and action_right:
            reward += 1.
        if not self._deterministic:  # Noisy rewards on the 'end' of chain.
            if self._row == self._size - 1 and self._column in [0, self._size - 1]:
                reward += self._rng.randn() * self._random_reward_scale

        # Transition dynamics
        if action_right:
            if self._deterministic or self._rng.rand() > self._random_dynamics:
                self._column = np.clip(self._column + 1, 0, self._size - 1)
            reward -= self._unscaled_move_cost / self._size
        else:
            self._column = np.clip(self._column - 1, 0, self._size - 1)
        self._row += 1

        observation = self._get_observation()
        done = False
        info = {}
        if self._row == self._size:
            done = True

        return observation, reward, done, info


if __name__ == "__main__":
    env = DeepSea()
    action_space = env.action_space
    obs_space = env.observation_space
    print(f"action_space: {action_space}, obs_space: {obs_space}")
    obs = env.reset()
    total_reward = 0
    while True:
        action = action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"total_reward: {total_reward}")
            break
