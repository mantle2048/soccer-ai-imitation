from grf_imitation.infrastructure.datas.buffer import ReplayBuffer
from grf_imitation.infrastructure.datas import Batch

import d4rl
import gym
from typing import Dict

def preprocess_d4rl_dataset(dataset: Dict) -> Dict:
    
    # rename the keys for simple styple
    if 'observations' in dataset.keys():
        dataset['obs'] = dataset.pop('observations')
    if 'actions' in dataset.keys():
        dataset['act'] = dataset.pop('actions')
    if 'rewards' in dataset.keys():
        dataset['rew'] = dataset.pop('rewards')
    if 'next_observations' in dataset.keys():
        dataset['next_obs'] = dataset.pop('next_observations')
    if 'terminals' in dataset.keys():
        dataset['terminated'] = dataset.pop('terminals')
    if 'timeouts' in dataset.keys():
        dataset['truncated'] = dataset.pop('timeouts')
        
    # remove dummy keys
    dummy_keys = [
        'infos/action_log_probs',
        'infos/qpos',
        'infos/qvel',
        'metadata/algorithm',
        'metadata/iteration',
        'metadata/policy/fc0/bias',
        'metadata/policy/fc0/weight',
        'metadata/policy/fc1/bias',
        'metadata/policy/fc1/weight',
        'metadata/policy/last_fc/bias',
        'metadata/policy/last_fc/weight',
        'metadata/policy/last_fc_log_std/bias',
        'metadata/policy/last_fc_log_std/weight',
        'metadata/policy/nonlinearity',
        'metadata/policy/output_distribution',
    ]
    for key in dummy_keys:
        if key in dataset.keys():
            dataset.pop(key)
    
    # set 'terminated' for 'done'
    done = dataset.pop('terminated') + dataset.pop('truncated')
    dataset['done'] = done
    return dataset

def load_d4rl_dataset(env_name: str) -> 'ReplayBuffer':
    expert_data_name =  '-'.join([env_name.split('-')[0].lower(), 'expert', 'v2'])
    print(f"Load d4rl dataset: {expert_data_name}")
    env = gym.make(expert_data_name)
    dataset = preprocess_d4rl_dataset(env.get_dataset())
    data = Batch(dataset)
    replay_buffer = ReplayBuffer.from_data(
        obs=dataset["obs"],
        act=dataset["act"],
        rew=dataset["rew"],
        done=dataset["done"],
        next_obs=dataset["next_obs"]
    )
    return replay_buffer
    

