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
        '1e-4',
        '--disc-update-num',
        '2',
        '--which-gpu',
        '0',
        '--expert',
        'football-bilibili-win-obs45',
        '--env-name',
        'malib-5vs5',
        '--seed',
        '1',
        '--n-itr',
        '1000',
        '--num-workers',
        '100',
        '--step-per-itr',
        '30000',
        '--buffer-size',
        '120000',
        '--repeat-per-itr',
        '10',
        '--batch-size',
        '12000',
        '--tabular-log-freq',
        '5',
        '--param-log-freq',
        '5',
        '--video-log-freq',
        '100',
        '--gamma',
        '0.999',
        '--gae-lambda',
        '0.98',
        '--lr',
        '3e-3',
        '--entropy-coeff',
        '0.01',
        '--layers',
        '64','64',
        '--activation',
        'tanh',
        '--opponent',
        'buildin',
        # '--lr-schedule',
        # 'Pi: [[0, 1.0], [1000, 0.1]]; \
        #  Disc: [[0, 1.0], [1000, 0.1]]',
        '--score-cut',
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




