# +
from grf_imitation.infrastructure.data.utils.load_dataset import load_d4rl_dataset

import d4rl
import gym
from typing import Dict

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

if __name__ == '__main__':
    env_name = 'halfcheetah-expert-v2'
    replay_buffer = load_d4rl_dataset(env_name)
    batch = replay_buffer.sample(128)
    import ipdb; ipdb.set_trace()




