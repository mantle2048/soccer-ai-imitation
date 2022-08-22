from grf_imitation.scripts.run_gail import get_parser, get_config, main
from grf_imitation.rl_trainer import RL_Trainer
from grf_imitation.algos import GAILAgent
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
        '--disc-lr',
        '1e-5',
        '--disc-update-num',
        '2',
        '--env-name',
        'HalfCheetah-v2',
        '--seed',
        '5',
        '--n-itr',
        '1000',
        '--num-workers',
        '10',
        '--step-per-itr',
        '1000',
        '--repeat-per-itr',
        '20',
        '--batch-size',
        '1000',
        '--tabular-log-freq',
        '20',
        '--param-log-freq',
        '50',
        '--gae-lambda',
        '0.95',
        '--lr',
        '3e-4',
        '--obs-norm',
        '--entropy-coeff',
        '0.001',
        '--lr-schedule',
        'Pi: [[0, 1.0], [1000, 0.1]]; \
         Disc: [[0, 1.0], [1000, 0.1]]',
        # '--recompute-adv',
        # '--ret-norm',
        # '--adv-norm',

    ])

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=agent_config['n_itr'])




