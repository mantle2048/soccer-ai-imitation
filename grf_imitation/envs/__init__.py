import gym
from gym.spaces import Box
from typing import Dict
from gym.wrappers import NormalizeObservation
from gfootball.env import create_environment

from grf_imitation.envs.wrappers import NetEase214Wrapper

def make_env(env_name: str='malib-5vs5', env_config: Dict={}):

    if env_name == 'malib-5vs5':
        env = create_environment(
            env_name='malib_5_vs_5',
            representation='raw',
            rewards='scoring',
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            action_set='v2',
            write_video=False,
            dump_frequency=1,
            logdir='',
            extra_players=None,
            number_of_left_players_agent_controls=4,
            number_of_right_players_agent_controls=4,
        )
    seed = env_config.get('seed', 0)
    env.seed(seed)
    env.action_space.seed(seed)

    env = NetEase214Wrapper(env, env_config)

    if isinstance(env.action_space, Box):
        env = ClipAction(env)
    if env_config.get('obs_norm'):
        env = NormalizeObservation(env)
    return env
