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
        '6e-6',
        '--disc-update-num',
        '2',
        '--env-name',
        'malib-5vs5',
        '--seed',
        '0',
        '--n-itr',
        '5000',
        '--num-workers',
        '30',
        '--step-per-itr',
        '3000',
        '--buffer-size',
        '3000',
        '--repeat-per-itr',
        '10',
        '--batch-size',
        '100',
        '--tabular-log-freq',
        '10',
        '--param-log-freq',
        '10',
        '--video-log-freq',
        '1000',
        '--gamma',
        '0.993',
        '--gae-lambda',
        '0.95',
        '--lr',
        '1e-4',
        '--entropy-coeff',
        '0.02',
        '--layers',
        '256','256', '256',
        '--activation',
        'relu',
        '--score-cut',
        '--opponent',
        'buildin',
        # '--lr-schedule',
        # 'Pi: [[0, 1.0], [1000, 0.1]]; \
        #  Disc: [[0, 1.0], [1000, 0.1]]',
        '--extra-buildin-act',
        # '--obs-norm',
        # '--ret-norm',
        # '--adv-norm',
        # '--recompute-adv',

    ])

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=agent_config['n_itr'])




