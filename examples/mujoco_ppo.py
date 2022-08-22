from grf_imitation.scripts.run_ppo import get_parser, get_config, main
from grf_imitation.rl_trainer import RL_Trainer
from grf_imitation.algos import PPOAgent
# import ray
# ray.init(
#     ignore_reinit_error=True,
#     local_mode=True,
# )
# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args([
        '--env-name',
        'CartPole-v1',
        '--seed',
        '3',
        '--n-itr',
        '10',
        '--num-workers',
        '2',
        '--step-per-itr',
        '2000',
        '--repeat-per-itr',
        '10',
        '--batch-size',
        '100',
        '--tabular-log-freq',
        '1',
        '--param-log-freq',
        '1',
        '--gae-lambda',
        '0.95',
        '--lr',
        '3e-4',
        '--obs-norm',
        '--ret-norm',
        '--recompute-adv'
        # '--adv-norm',

    ])

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=agent_config['n_itr'])




