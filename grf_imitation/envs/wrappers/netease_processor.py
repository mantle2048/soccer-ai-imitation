import gym
from gym.spaces import Box
import numpy as np
from typing import Dict, Union, List

from grf_imitation.envs.feature_encoder import MyFeatureEncoder

class NetEaseWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, env_config: Dict):
        super().__init__(env)

        # observation process
        self.observation_space =  \
            Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)
        self.feature_encoder = MyFeatureEncoder()

        # action process
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
        self.max_episode_steps = 3000
        self._step = 0
        self._game_mode = 0

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        self._step = 0
        return self.observation(obs)

    def step(self, act):
        act = self.action(act)
        obs, rew, done, info = self.env.step(act)
        if self._game_mode != obs[0]['game_mode']:
            info['game_mode_changed'] = True
        else:
            info['game_mode_changed'] = False
        self._game_mode = obs[0]['game_mode']
        self._step += 1

        # Maximum episode length
        if self._step >= self.max_episode_steps:
            done = True
        return self.observation(obs), rew, done, info

    def action(self, action):
        act = action.flatten()[:4]
        if self.opponent == 'buildin':
            oppo_act = np.array([19, 19, 19, 19])
        elif self.opponent == 'random':
            oppo_act = np.random.randint(0, 19, (4,))
        elif self.opponent == 'selfplay':
            oppo_act = action.flatten()[4:]
        else:
            # static opponent
            oppo_act = np.array([0, 0, 0, 0])
        return np.concatenate([act, oppo_act])

    def observation(self, observation: List[Dict]) -> np.ndarray:
        obs_list = []
        for idx, obs_raw in enumerate(observation):
            obs_dict = self.feature_encoder.encoder('simple', obs_raw, idx)[0]
            obs = np.hstack([np.array(obs_dict[k], dtype=np.float32).flatten() for k in obs_dict])
            obs_list.append(obs)
        return np.array(obs_list)
