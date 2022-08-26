import numpy as np
import torch
from typing import Dict, Any, List
from torch.nn import functional as F
from torch import optim

from grf_imitation.infrastructure.utils import pytorch_util as ptu
from grf_imitation.policies.ppo_policy import PPOPolicy
from grf_imitation.infrastructure.datas import Batch

class GAILPolicy(PPOPolicy):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.disc_net =  \
            ptu.build_mlp(
                input_size=self.obs_dim,
                output_size=self.act_dim,
                layers=self.layers,
            )
        self.disc_net.to(ptu.device)
        self.disc_net.apply(ptu.init_weights)
        ptu.scale_last_layer(self.disc_net)
        self.disc_optimizer = optim.Adam(
            self.disc_net.parameters(),
            self.config.get('disc_lr'),
        )
        self.optimizers.update(Disc=self.disc_optimizer)

    def update_disc(
        self,
        pi_batch: Batch = None,
        exp_batch: Batch = None,
        **kwargs: Any
    )-> Dict[str, float]:

        exp_obss = ptu.from_numpy(exp_batch.obs)
        exp_acts = ptu.from_numpy(exp_batch.act)
        pi_obss = ptu.from_numpy(pi_batch.obs)
        pi_acts = ptu.from_numpy(pi_batch.act)

        score_pi = self._score(pi_obss, pi_acts)
        score_exp = self._score(exp_obss, exp_acts)
        loss_pi = -F.logsigmoid(-score_pi).mean()
        loss_exp = -F.logsigmoid(score_exp).mean()
        loss_disc = loss_pi + loss_exp
        self.disc_optimizer.zero_grad()
        loss_disc.backward()
        self.disc_optimizer.step()

        train_log = {}
        train_log['Discriminator loss'] = ptu.to_numpy(loss_disc)
        train_log['Acc pi'] = ptu.to_numpy((score_pi < 0).float().mean())
        train_log['Acc exp'] = ptu.to_numpy((score_exp > 0).float().mean())
        return train_log

    def run_disc_prediction(
        self, obs: np.ndarray, act: np.ndarray, 
    ):

        obs = ptu.from_numpy(obs)
        act = ptu.from_numpy(act)
        scores = self._score(obs, act)
        predictions = -F.logsigmoid(-scores)
        return ptu.to_numpy(predictions)[:, 0]

    def _score(self, obs: torch.Tensor, act: torch.Tensor):
        return torch.gather(
            self.disc_net(obs),
            dim=1,
            index=act.long().view(-1, 1)
        )
