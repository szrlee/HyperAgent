from typing import Any

import gym
import numpy as np
import cloudpickle
from .atari_wrapper import wrap_atari

class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)

class NoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_dim, noise_std=1.):
        super().__init__(env)
        assert noise_dim > 0
        self.env = env
        self.noise_dim = noise_dim
        self.noise_std = noise_std

    def reset(self):
        state = self.env.reset()
        self.now_noise = np.random.normal(0, 1, self.noise_dim) * self.noise_std
        return np.hstack([self.now_noise, state])

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return np.hstack([self.now_noise, state]), reward, done, info

def make_env(env_name, max_step=None, env_config={}):
    env = gym.make(env_name, **env_config)
    if max_step is not None:
        env._max_episode_steps = max_step
    return env

def make_atari_env(args):
    return wrap_atari(args.task, frame_stack=args.frames_stack, episode_life=True, clip_rewards=True, atari_100k=args.atari_100k)

def make_atari_env_watch(args):
    return wrap_atari(args.task, frame_stack=args.frames_stack, episode_life=False, clip_rewards=False,
                      max_episode_steps=int(27e3), atari_100k=args.atari_100k)
