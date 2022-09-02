import numpy as np
import ray
import warnings
from typing import Optional, Union, List, Dict
from ray.actor import ActorHandle
from grf_imitation.infrastructure.datas import Batch

def synchronous_parallel_sample(
    *,
    remote_workers: List[ActorHandle],
    max_steps: Optional[int] = None,
    max_episodes: Optional[int] = None,
    concat: bool = False
) -> List[Batch]:
    """
    Runs parallel and synchronous rollouts on all remote workers.
    Waits for all workers to return from the remote calls.
    If no remote workers exist (num_workers == 0), use the local worker
    for sampling.

    Args:
        worker_set: The WorkerSet to use for sampling.
        max_steps: Optional number of steps to be included in the sampled batch.
        concat: Whether to concat all resulting batches at the end and return the
            concated batch.
    Returns:
        The list of collected sample batch (one for each parallel
        rollout worker in the given `worker_set` if no max_steps).
    """
    batch_list = []
    if not max_steps and not max_episodes:
        # Loop over remote workers' `sample()` method in parallel.
        batches = ray.get(
            [worker.sample.remote() for worker in remote_workers]
        )
        batch_list.extend(batches)

    # max_steps first, then max_episodes
    elif max_steps:
        steps = 0
        worker_step = int(np.round(max_steps / len(remote_workers)))
        while steps < max_steps:
            batches = ray.get(
                [worker.sample.remote(sample_step=worker_step) for worker in remote_workers]
            )
            for batch in batches:
                surplus_steps = max_steps - steps
                if surplus_steps < len(batch):
                    batch = batch[:surplus_steps]
                    batch.truncated[-1] = True
                batch_list.append(batch)
                steps += len(batch)
                if steps >= max_steps: break

    elif max_episodes:
        episodes = 0
        while episodes < max_episodes:
            surplus_episodes = max_episodes - episodes
            remote_workers = remote_workers[:surplus_episodes]
            batches = ray.get(
                [worker.sample.remote() for worker in remote_workers]
            )
            for batch in batches:
                batch_list.append(batch)
                episodes += 1

    if concat is True:
        batch_full = Batch.cat(batch_list)
        return batch_full
    else:
        return batch_list
