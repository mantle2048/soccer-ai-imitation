import gym
from gym.spaces import Box
import numpy as np
from typing import Dict, Union, List

from grf_imitation.envs.feature_encoder import MyFeatureEncoder

class NetEase214Wrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, env_config: Dict):
        super().__init__(env)

        # observation process
        self.observation_space =  \
            Box(low=-np.inf, high=np.inf, shape=(214,), dtype=np.float32)
        self.feature_encoder = MyFeatureEncoder()

        # action process
        if env_config.get('extra_buildin_act'):
            self.action_space = gym.spaces.Discrete(20) # add build-ai action
        else:
            self.action_space = gym.spaces.Discrete(19) # remove build-ai action

        # choose the opponent
        self.opponent = env_config.get('opponent')
        assert self.opponent in (
            'buildin',
            'random',
            'static',
            'selfplay',
        ), f"Not Supported AI type: {self.opponent}"

        # step set
        self.max_episode_steps = env_config.get('ep_len')
        self._step = 0

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        self._step = 0
        return self.observation(obs)

    def step(self, act):
        act = self.action(act)
        obs, rew, done, info = self.env.step(act)
        self._step += 1

        # Maximum episode length
        if self._step == self.max_episode_steps: done = True
        return self.observation(obs), rew[:4], done, info

    def action(self, action):
        act = action.flatten()[:4]
        if self.opponent == 'buildin':
            oppo_act = np.array([19, 19, 19, 19])
        elif self.opponent == 'random':
            oppo_act = np.random.randint(0, 19, (4,))
        elif self.opponent == 'selfplay':
            oppo_act = act.copy()
        else:
            # static opponent
            oppo_act = np.array([0, 0, 0, 0])
        return np.concatenate([act, oppo_act])

    def observation(self, observation: List[Dict]) -> np.ndarray:
        # remove opponent obs
        observation = observation[:4]
        obs_list = []
        for obs_raw in observation:
            obs_dict = self.feature_encoder.encode(obs_raw)[0]
            obs = np.hstack([np.array(obs_dict[k], dtype=np.float32).flatten() for k in obs_dict])
            obs_list.append(obs)
        return np.array(obs_list)
