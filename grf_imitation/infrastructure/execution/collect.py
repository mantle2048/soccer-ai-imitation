import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from grf_imitation.infrastructure.datas import Batch
from grf_imitation.infrastructure.execution import synchronous_parallel_sample, WorkerSet 

def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # If num_episodes is set, stop if limit reached.
    if num_episodes and episodes >= num_episodes:
        return False
    # If num_steps is set, stop if limit reached.
    elif num_steps and steps >= num_steps:
        return False
    # Otherwise, keep going.
    return True

def collect(
    agent,
    num_episodes: int=None,
    num_steps: int=None,
) -> List[Batch]:

    assert hasattr(agent, 'workers') \
        and isinstance(agent.workers, WorkerSet), \
        f'Agent: {agent} must have workers to collect.'

    train_batch_list = []
    if agent.workers.remote_workers():
        batches = synchronous_parallel_sample(
            remote_workers=agent.workers.remote_workers(),
            max_steps=num_steps,
            max_episodes=num_episodes,
            concat=False,
        )
        train_batch_list.extend(batches)
        return train_batch_list

    else:
        local_worker = agent.workers.local_worker()
        env, policy = local_worker.env, local_worker.policy

        steps = 0
        episodes = 0
        train_batch_list = []
        while keep_going(steps, num_steps, episodes, num_episodes):
            batch = local_worker.sample()
            surplus_steps = num_steps - steps
            if surplus_steps < len(batch):
                batch = batch[:surplus_steps]
                batch.truncated[-1] = True
            steps += len(batch)
            episodes += 1
            train_batch_list.append(batch)

    return train_batch_list

