import os.path as osp
import numpy as np

from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from typing import Dict,Union,List,Tuple

from grf_imitation import user_config as conf
from grf_imitation.envs import make_env
from grf_imitation.infrastructure.execution import WorkerSet
from grf_imitation.algos.ppo import PPOAgent
from grf_imitation.infrastructure.datas import ReplayBuffer, Batch
from grf_imitation.infrastructure.datas.utils.load_dataset import load_grf_dataset

class GAILAgent(PPOAgent):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.expert_buffer = load_grf_dataset(config.get('expert'))
        print("Expert Buffer: ", end='')
        print("obs: ", self.expert_buffer.obs.shape, end='')
        print("act: ", self.expert_buffer.act.shape)
        self.disc_update_num = config.setdefault('disc_update_num', 4)

    def process_fn(self, batch: Union[List[Batch], Batch]) -> Batch:
        if isinstance(batch, List):
            batch = Batch.cat(batch)
        batch = self._football_process_fn(batch)
        batch = self.get_rew(batch)
        batch = super().process_fn(batch)
        return batch
    def train(self, batch_size: int, repeat: int) -> Dict:

        disc_train_log = {}
        pi_batch = self.sample(0)
        disc_batch_size = len(pi_batch) // self.disc_update_num
        for pi_minibatch in \
                pi_batch.split(disc_batch_size, merge_last=True):
            exp_minibatch = self.expert_buffer.sample(disc_batch_size)
            disc_train_log = \
                self.policy.update_disc(pi_minibatch, exp_minibatch)
        pi_train_log = super().train(batch_size, repeat)
        train_log = {**disc_train_log, **pi_train_log}
        return train_log

    def get_rew(self, batch: Batch):
        obss, acts = batch.obs, batch.act
        batch['rew'] = self.policy.run_disc_prediction(obss, acts)
        return batch

    def _football_process_fn(self, batch: Batch) -> Batch:
        for k in batch.keys():
            if k in ('obs', 'next_obs'):
                batch[k] = batch[k].transpose(1, 0, 2).reshape(-1, 214)
            elif k in ('act', 'rew', 'ep_rew'):
                batch[k] = batch[k].T.flatten()
            elif k in ('done', 'eps_id', 'ep_len', 'terminal'):
                batch[k] = np.tile(batch[k], 4)
        # Score of our side or opponent side, cut the rollout, but not reset
        indices = np.nonzero(batch.rew)
        batch.done[indices] = True
        batch.terminal[indices] = True
        return batch
