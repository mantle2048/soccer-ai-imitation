import gym
import os
import os.path as osp
import urllib.request
import h5py
from tqdm import tqdm


def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)

set_dataset_path(conf.LOCAL_DATASET_DIR)

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    if dataset_url is None:
        return 'No dataset url!'
    _, dataset_name = osp.split(dataset_url)
    dataset_filepath = osp.join(DATASET_PATH, dataset_name)
    return dataset_filepath

def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not osp.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not osp.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath

class OfflineEnv(gym.Env):
    """
    Base class for offline RL envs.
    Args:
        dataset_url: URL pointing to the dataset.
    """
    def __init__(self, dataset_url=None, **kwargs):
        super(OfflineEnv, self).__init__(**kwargs)
        self.dataset_url = dataset_url

    @property
    def dataset_filepath(self):
        return filepath_from_url(self.dataset_url)

    def get_dataset(self, h5path=None):
        if h5path is None:
            if self.dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in data_dict:
            assert key in ['obs', 'act', 'rew', 'done'], 'Dataset is missing key %s' % key

        N_samples = data_dict['obs'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['obs'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['obs'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['act'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['act'].shape[1:]), str(self.action_space.shape))
        if data_dict['rew'].shape == (N_samples, 1):
            data_dict['rew'] = data_dict['rew'][:, 0]
        assert data_dict['rew'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rew'].shape))
        if data_dict['done'].shape == (N_samples, 1):
            data_dict['done'] = data_dict['done'][:, 0]
        assert data_dict['done'].shape == (N_samples,), 'done has wrong shape: %s' % (
            str(data_dict['rew'].shape))
        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path=None):
        """
        Returns a slice of the full dataset.
        Args:
            chunk_id (int): An integer representing which slice of the dataset to return.
        Returns:
            A dictionary containing obs, act, rew, and done.
        """
        if h5path is None:
            if self.dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        dataset_file = h5py.File(h5path, 'r')

        if 'virtual' not in dataset_file.keys():
            raise ValueError('Dataset is not a chunked dataset')
        available_chunks = [int(_chunk) for _chunk in list(dataset_file['virtual'].keys())]
        if chunk_id not in available_chunks:
            raise ValueError('Chunk id not found: %d. Available chunks: %s' % (chunk_id, str(available_chunks)))

        load_keys = ['obs', 'act', 'rew', 'done']
        data_dict = {k: dataset_file['virtual/%d/%s' % (chunk_id, k)][:] for k in load_keys}
        dataset_file.close()
        return data_dict
