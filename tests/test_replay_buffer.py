# +
import numpy as np
from grf_imitation.infrastructure.datas.utils.load_dataset import load_d4rl_dataset, preprocess_d4rl_dataset
from grf_imitation.infrastructure.datas import Batch, convert_batch_to_dict
from grf_imitation.infrastructure.datas.buffer import ReplayBuffer
import pickle
import torch

import d4rl
import gym
from typing import Dict

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

replaybuffer = load_d4rl_dataset("halfcheetah-expert-v2")

dataset = preprocess_d4rl_dataset(gym.make("halfcheetah-expert-v2").get_dataset())

data_dict = {}
for key, value in dataset.items():
    data_dict[key] = value[0:1000]
batch = Batch(data_dict)
batch['done'].sum()

replaybuffer.add_batch(batch)

replaybuffer.sample(16)
