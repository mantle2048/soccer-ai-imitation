import h5py
import os.path as osp

from grf_imitation import user_config as conf
from grf_imitation.infrastructure.datas.buffer import ReplayBuffer
from grf_imitation.infrastructure.datas.utils.converter import from_hdf5

def load_grf_dataset(expert: str) -> 'ReplayBuffer':
    dataset_path = osp.join(conf.LOCAL_DATASET_DIR, expert + '.hdf5')
    print(f'Loading dateset from {dataset_path}')
    with h5py.File(dataset_path, 'r') as f:
        data_dict = from_hdf5(f)
    return ReplayBuffer.from_data(
        obs=data_dict['obs'][:2400000], act=data_dict['act'][:2400000]
    )
