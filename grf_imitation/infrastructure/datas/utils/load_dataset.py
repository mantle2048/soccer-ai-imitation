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
    data_dict['obs'] = data_dict['obs'][:150000]
    data_dict['act'] = data_dict['act'][:150000]
    return ReplayBuffer.from_data(
        obs=data_dict['obs'].transpose(1, 0, 2).reshape(-1, 214),
        act=data_dict['act'].T.flatten()
    )
