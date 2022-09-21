import numpy as np
import itertools
import torch
from typing import Any, Dict, List, Optional
from torch import nn
from torch import optim
from torch import distributions
from torch import optim

from grf_imitation.infrastructure.datas import Batch
from grf_imitation.infrastructure.utils import pytorch_util as ptu

class OnPolicy(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        # init params
        self.config = config

        self.obs_dim = config['obs_dim']
        self.act_dim = config['act_dim']
        self.layers = config['layers']
        self.discrete = config['discrete']
        self.lr = config['lr']
        self.entropy_coeff = config['entropy_coeff']
        self.grad_clip = config['grad_clip']
        self.epsilon = config['epsilon']
        self.activation = config['activation']
        self.optimizers = {}

        # discrete or continus
        if self.discrete:
            self.logits_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers,
                                            activation=self.activation
                                            )
            self.logits_net.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(
                params = self.logits_net.parameters(),
                lr = self.lr)
        else:
            self.logits_net = None
            self.mean_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers,
                                            activation=self.activation
                                          )
            self.logstd = nn.Parameter(
                    -0.5 * torch.ones(self.act_dim, dtype=torch.float32, device=ptu.device))

            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                params = itertools.chain(self.mean_net.parameters(),[self.logstd]),
                lr = self.lr)
        self.optimizers.update(Pi=self.optimizer)

        # init baseline
        self.baseline = ptu.build_mlp(
            input_size=self.obs_dim,
            output_size=1,
            activation=self.activation,
            layers=self.layers,
        )
        self.baseline.to(ptu.device)
        self.baseline_optimizer = optim.Adam(
            self.baseline.parameters(),
            self.lr,
        )
        self.optimizers.update(Baseline=self.baseline_optimizer)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        '''
            query the policy with observation(s) to get selected action(s)
        '''
        if len(obs.shape) == 1:
            obs = obs[None]

        obs = ptu.from_numpy(obs.astype(np.float32))

        act_dist = self.forward(obs)

        act = act_dist.sample()

        act = act.squeeze()
        # if self.discrete and act.shape != ():
        #     act = act.squeeze()

        return ptu.to_numpy(act)

    def update(
        self,
        batch: Batch = None,
        **kwargs: Any
    )-> Dict[str, float]:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor):
        '''
        This function defines the forward pass of the network.
        You can return anything you want, but you should be able to differentiate
        through it. For example, you can return a torch.FloatTensor. You can also
        return more flexible objects, such as a
        `torch.distributions.Distribution` object. It's up to you!
        '''
        if self.discrete:
            logits_na = self.logits_net(obs)
            act_dist = distributions.Categorical(logits=logits_na)

        else:
            mean_na = self.mean_net(obs)
            std_na = torch.exp(self.logstd)
            act_dist = distributions.MultivariateNormal(loc=mean_na, scale_tril=torch.diag(std_na))
            # helpful: difference between multivariatenormal and normal sample/batch/event shapes:
            # https://bochang.me/blog/posts/pytorch-distributions/
            # https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/

        return act_dist

    def run_baseline_prediction(self, obs: np.ndarray):

        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array
            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]
        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions[:, 0])

    def save(self, filepath=None):
        torch.save(self.state_dict(), filepath)

    def set_weights(self, weights: Dict):
        self.load_state_dict(weights)

    def get_weights(self) -> Dict:
        return {k: v.cpu().detach() for k, v in self.state_dict().items()}

    def _get_log_prob(self, obss, acts):
        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        act_dist = self.forward(obss)
        return ptu.to_numpy(act_dist.log_prob(acts))

