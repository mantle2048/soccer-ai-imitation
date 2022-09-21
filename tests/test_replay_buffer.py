# +
import numpy as np
from grf_imitation.infrastructure.datas import Batch, convert_batch_to_dict
from grf_imitation.infrastructure.datas.buffer import ReplayBuffer
# from tianshou.data.buffer.base import ReplayBuffer
from grf_imitation import user_config as conf
from grf_imitation.infrastructure.datas.utils.converter import from_hdf5, to_hdf5
import pickle
import torch
import h5py
import os.path as osp

import gym
from typing import Dict

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

expert = 'football-bilibili-win'
dataset_path = osp.join(conf.LOCAL_DATASET_DIR, expert + '.hdf5')
test_path = osp.join(conf.LOCAL_DATASET_DIR, 'test.hdf5')
print(dataset_path)
with h5py.File(dataset_path, "r") as f:
    data_dict = from_hdf5(f)
    data_dict['obs'] = data_dict['obs'][:3000]
    data_dict['act'] = data_dict['act'][:3000]
with h5py.File(test_path, "w") as f:
    to_hdf5(data_dict, f)
# expert_buffer = ReplayBuffer.load_hdf5(dataset_path)
# expert_buffer.sample(10)

expert_buffer
