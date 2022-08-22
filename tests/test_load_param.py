from grf_imitation import user_config as conf
import os
import os.path as osp
# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook

data_dir = conf.LOCAL_LOG_DIR


def test_shelve_param():
    import pickle
    import shelve
    run_dir = osp.join(data_dir, 'ppo_CartPole-v1', 'ppo_CartPole-v1_3')
    param_dir = osp.join(run_dir, 'params')
    shelf = osp.join(param_dir, 'params')
    k = shelve.open(shelf)
    print(list(k.keys()))


test_shelve_param()




