import random
import numpy as np
import torch
import os
import warnings
import ray
from typing import Optional, List, Dict, Union, Callable, overload
from grf_imitation.infrastructure.execution import RolloutSaver
from grf_imitation.infrastructure.datas import Batch
from grf_imitation.infrastructure.utils.gym_util import get_max_episode_steps

def update_gloabl_seed(
    seed: Optional[int] = None,
    worker_id: int=0,
) -> None:
    """Seed global modules such as random, numpy, torch.
    This is useful for debugging and testing.
    Argsw
        seed: An optional int seed. If None, will not do
            anything.
    """
    if seed is None:
        return

    computed_seed: int = worker_id * 1000 + seed
    # Python random module.
    random.seed(computed_seed)
    # Numpy.
    np.random.seed(computed_seed)
    # Torch.
    torch.manual_seed(computed_seed)

def update_env_seed(
    env,
    seed: Optional[int] = None,
    worker_id: int=0,
):
    """Set a deterministic random seed on environment.
    NOTE: this may not work with remote environments (issue #18154).
    """
    if not seed:
        return

    # A single RL job is unlikely to have more than 10K
    # rollout workers.
    computed_seed: int = worker_id * 1000 + seed

    # Gym.env.
    # This will silently fail for most OpenAI gyms
    # (they do nothing and return None per default)
    if not hasattr(env, "seed"):
        warnings.wran("Env doesn't support env.seed(): {}".format(env))
    else:
        env.seed(computed_seed)

class RolloutWorker:

    """Common experience collection class.
    This class wraps a policy instance and an environment class to
    collect experiences from the environment. You can create many replicas of
    this class as Ray actors to scale RL training."""

    @classmethod
    def as_remote(
        cls,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[Union[int, float]] = None,
        memory: Optional[int] = None,
        object_store_memory: Optional[int] = None,
        resources: Optional[dict] = None,
    ) -> type:
        """Returns RolloutWorker class as a `@ray.remote using given options`.
        The returned class can then be used to instantiate ray actors.
        Args:
            num_cpus: The number of CPUs to allocate for the remote actor.
            num_gpus: The number of GPUs to allocate for the remote actor.
                This could be a fraction as well.
            memory: The heap memory request for the remote actor.
            object_store_memory: The object store memory for the remote actor.
            resources: The default custom resources to allocate for the remote
                actor.
        Returns:
            The `@ray.remote` decorated RolloutWorker class.
        """
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def __init__(
        self,
        *,
        worker_id: int = 0,
        env_maker: Callable[["env_name", "env_config"], "env"],
        policy_maker: Callable[["policy_name", "policy_config"], "policy"],
        config: Dict,
    ):
        """
        Initializes a RolloutWorker instance.

        Args:
            env_creator: Function that returns a gym.Env given an 
                wrapped configuration.
            worker_id: For remote workers, this should be set to a
                non-zero and unique value. This id is passed to created envs
                through EnvContext so that envs can be configured per worker.
            config: Config to pass to worker (consists of env_config,
                policy_config)
        """
        self.worker_id = worker_id
        self.config = config
        update_gloabl_seed(config.get('seed'), worker_id)
        self.policy = policy_maker(
            policy_name = config.get('policy_name'),
            policy_config = config.get('policy_config')
            )
        # env for sample steps for collect
        self.env = env_maker(
            env_name = config.get('env_name'),
            env_config = config.get('env_config')
        )
        # env for sample episode for evalute
        self.eval_env = env_maker(
            env_name = config.get('env_name'),
            env_config = config.get('env_config')
        )
        # share the obs norm rms
        if self.config.get('obs_norm'):
            self.eval_env.obs_rms = self.env.obs_rms

        update_env_seed(self.env, config.get('seed'), worker_id)
        update_env_seed(self.eval_env, config.get('seed'), worker_id)
        self.max_step = get_max_episode_steps(self.env)
        self.saver = RolloutSaver(save_info=True)

        self._cur_eplen = 0
        self._cur_eprew = 0.
        self._cur_obs = self.env.reset()

    def sample(self, sample_step: int=None) -> Batch:

        if sample_step:
            return self.sample_steps(sample_step)
        else:
            return self.sample_episode()

    def sample_steps(self, sample_step: int) -> Batch:
        step = 0
        step_list = []
        env, policy = self.env, self.policy
        while step < sample_step:
            terminal = False
            act = policy.get_action(self._cur_obs)
            next_obs, rew, done, info = env.step(act)
            self._cur_eprew += rew
            self._cur_eplen += 1
            if done and self._cur_eplen != self.max_step:
                terminal = True
            step_return = Batch(
                obs=self._cur_obs, act=act, next_obs=next_obs,
                rew=rew, done=done, info=info, terminal=terminal,
                ep_len=self._cur_eplen, ep_rew=self._cur_eprew,
            )
            step_list.append(step_return)
            if done:
                self._cur_obs = env.reset()
                self._cur_eplen = 0
                self._cur_eprew = 0.
            else:
                self._cur_obs = next_obs
            step += 1
        batch = Batch.stack(step_list)
        return batch

    def sample_episode(self) -> Batch:
        step_list = []
        env, policy = self.eval_env, self.policy
        ep_rew, ep_len, step_list = 0.0, 0, []
        done, terminal = False, False
        obs = env.reset()
        while not done:
            act = policy.get_action(obs)
            next_obs, rew, done, info = env.step(act)
            ep_rew += rew
            ep_len += 1
            if done and ep_len != self.max_step:
                terminal = True
            step_return = Batch(
                obs=obs, act=act, next_obs=next_obs,
                rew=rew, done=done, info=info,
                ep_len=ep_len, ep_rew=ep_rew,
                terminal=terminal
            )
            step_list.append(step_return)
            obs = next_obs
        batch = Batch.stack(step_list)
        return batch

    def set_weights(self, weights):
        self.policy.set_weights(weights)

    def get_weights(self):
        return self.policy.get_weights()

    def get_statistics(self):
        statistics = {}
        if hasattr(self.env, 'obs_rms'):
            statistics['obs_mean'] = self.env.obs_rms.mean
            statistics['obs_var'] = self.env.obs_rms.var
            statistics['count'] = self.env.obs_rms.count
        return statistics

    def set_statistics(self, statistics: Dict):
        if not statistics: return
        if hasattr(self.env, 'obs_rms'):
            self.env.obs_rms.mean = statistics['obs_mean']
            self.env.obs_rms.var = statistics['obs_var']
            self.env.obs_rms.count = statistics['count']

    def stop(self):
        """Releases all resources used by this RolloutWorker."""
        self.env.close()

    def get_obs_statistics(self):
        return (self.env.obs_rms.mean, self.env.obs_rms.var)
