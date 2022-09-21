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
        '3e-4',
        '--disc-update-num',
        '2',
        '--which-gpu',
        '0',
        '--expert',
        'football-bilibili-win-simple',
        '--env-name',
        'malib-5vs5',
        '--seed',
        '2',
        '--n-itr',
        '30000',
        '--num-workers',
        '10',
        '--step-per-itr',
        '1000',
        '--buffer-size',
        '8000',
        '--repeat-per-itr',
        '20',
        '--batch-size',
        '800',
        '--tabular-log-freq',
        '10',
        '--param-log-freq',
        '1000',
        '--video-log-freq',
        '1000',
        '--gamma',
        '0.993',
        '--gae-lambda',
        '0.98',
        '--lr',
        '1e-3',
        '--entropy-coeff',
        '0.01',
        '--layers',
        '64','64',
        '--activation',
        'tanh',
        '--opponent',
        'selfplay',
        '--score-cut',
        '--lr-schedule',
        'Pi: [[0, 1.0], [10000, 0.1]]; \
         Disc: [[0, 1.0], [10000, 0.1]]',
        # '--extra-buildin-act',
        # '--obs-norm',
        '--ret-norm',
        '--adv-norm',
        # '--recompute-adv',

    ])

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=agent_config['n_itr'])
