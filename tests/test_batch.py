# +
import numpy as np
import argparse
from grf_imitation.infrastructure.datas import Batch, convert_batch_to_dict
import pickle
import torch

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -


def rollout():
    batch0 = Batch(img_obs=[np.zeros((5, 5, 3))])
    batch1 = Batch(img_obs=[np.zeros((5, 5, 3))])
    batch2 = Batch(img_obs=[])
    batch3 = Batch(img_obs=[])
    batch4 = Batch(img_obs=[])
    print(Batch.cat([batch0, batch1]).shape)
    empty_batch = Batch.cat([batch2, batch3, batch4])
    if not empty_batch.img_obs: print("132")
rollout()

if __name__ == '__main__':
    batch = Batch()
    batch['a'] = np.ones((2,3))
    batch['b'] = Batch(c=np.ones((2,3)), d=np.ones((2,3)))
    batch['c'] = Batch(e=Batch(f=np.ones((2,3))), g=Batch(h=np.ones((2,3))))
    from pprint import pprint as pp
    pp(batch)
    pp(convert_batch_to_dict(batch))

b = Batch(info = Batch(pp=np.ones(10)))
b = Batch.stack([b,b])
b
