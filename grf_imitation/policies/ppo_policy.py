import torch
import numpy as np

from typing import Any, Dict, List, Optional
from torch import nn
from torch.nn import functional as F

from .base_policy import OnPolicy
from grf_imitation.infrastructure.datas import Batch
from grf_imitation.infrastructure.utils import pytorch_util as ptu

class PPOPolicy(OnPolicy):

    def __init__(self, config: Dict):
        super().__init__(config)
        # init params
        self.epsilon = config['epsilon']

        self.apply(ptu.init_weights)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        ptu.scale_last_layer(self.logits_net if self.logits_net else self.mean_net)

    def update(
        self,
        batch: Batch = None,
        **kwargs: Any
    )-> Dict[str, float]:
        '''
            Update the policy using ppo-clip surrogate object
        '''
        obss = ptu.from_numpy(batch.obs)
        acts = ptu.from_numpy(batch.act)
        log_pi_old = ptu.from_numpy(batch.log_prob)
        advs = ptu.from_numpy(batch.adv)

        act_dist = self.forward(obss)
        log_pi = act_dist.log_prob(acts)
        entropy = act_dist.entropy().mean()

        ratio = torch.exp(log_pi - log_pi_old)
        surr1 = ratio * advs
        surr2 = ratio.clamp(
            1.0-self.epsilon, 1.0+self.epsilon
        ) * advs
        surrogate_obj = torch.min(surr1, surr2)
        loss = -torch.mean(surrogate_obj) - self.entropy_coeff * entropy
        # Userful extral info
        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        log_ratio = log_pi - log_pi_old
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipped = ratio.gt(1+self.epsilon) | ratio.lt(1-self.epsilon)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean()

        # optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        targets = ptu.from_numpy(batch.returns)
        baseline_preds = self.baseline(obss).flatten()

        ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
        ## [ N ] versus shape [ N x 1 ]
        ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
        assert baseline_preds.shape == targets.shape

        ## compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
        ## HINT: use `F.mse_loss`
        baseline_loss = F.mse_loss(baseline_preds, targets)

        # optimize `baseline_loss` using `self.baseline_optimizer`
        # HINT: remember to `zero_grad` first
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

        train_log = {}
        train_log['Training loss'] = ptu.to_numpy(loss)
        train_log['Entropy'] = ptu.to_numpy(entropy)
        train_log['KL Divergence'] = ptu.to_numpy(approx_kl)
        train_log['Clip Frac'] = ptu.to_numpy(clipfrac)
        train_log['Baseline loss'] = ptu.to_numpy(baseline_loss)
        return train_log
