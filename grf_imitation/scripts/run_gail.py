import argparse
import json

from typing import Dict
from grf_imitation.rl_trainer import RL_Trainer
from grf_imitation.algos import GAILAgent

def get_parser():
    parser = argparse.ArgumentParser()

    # GAIL config
    parser.add_argument('--disc-lr', type=float, default=2.5e-5)
    parser.add_argument('--disc-update-num', type=int, default=2)
    parser.add_argument('--expert', type=str, default='football-bilibili-win')

    # GRF config
    parser.add_argument('--opponent', type=str, default='buildin')
    parser.add_argument('--score-cut', action='store_true')

    # PPO config
    parser.add_argument('--buffer-size', type=int, default=5000)
    parser.add_argument('--env-name', type=str, default='CartPole-v1')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--which-gpu', default=0)
    parser.add_argument('--snapshot-mode', type=str, default="last")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-itr', '-n', type=int, default=100)
    parser.add_argument('--step-per-itr', type=int, default=5000) #steps collected per train iteration
    parser.add_argument('--repeat-per-itr', type=int, default=5) #steps collected per train iteration
    parser.add_argument('--batch-size', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--tabular-log-freq', type=int, default=1)
    parser.add_argument('--video-log-freq', type=int, default=None)
    parser.add_argument('--param-log-freq', type=int, default=10)
    parser.add_argument('--obs-norm', action='store_true')
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coeff', type=float, default=0.)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=[64,64])
    parser.add_argument('--activation', type=str, default="tanh")
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument("--recompute-adv", action='store_true')
    parser.add_argument("--ret-norm", action='store_true')
    parser.add_argument("--adv-norm", action='store_true')
    parser.add_argument(
        '--lr-schedule',
        default={},
        type=lambda x: {k.strip():json.loads(v) for k,v in (i.split(':') for i in x.split(';'))},
        help='lr schedule i.e. "--lr-schedule a: [[500, 0.1]]; b: [[500, 0.001]]" '
    )
    return parser

def get_config(args: argparse.Namespace) -> Dict:
    #####################
    ## SET AGENT CONFIGS
    #####################
    args.policy_name = 'gail'

    # policy args
    policy_config = {
        'policy_name': args.policy_name,
        'layers': args.layers,
        'activation': args.activation,
        'lr': args.lr,
        'disc_lr': args.disc_lr,
        'lr_schedule': args.lr_schedule,
        'epsilon': args.epsilon,
        'entropy_coeff': args.entropy_coeff,
        'grad_clip': args.grad_clip,
    }

    # logger args
    logger_config = {
        'exp_prefix': f"{args.policy_name}_{args.env_name}",
        'seed': args.seed,
        'snapshot_mode': args.snapshot_mode,
    }

    # env args
    env_config = {
        'env_name': args.env_name,
        'obs_norm': args.obs_norm,
        'seed': args.seed,
        'opponent': args.opponent,
    }

    agent_config = vars(args)

    agent_config.update(
        agent_class=GAILAgent,
        policy_config=policy_config,
        env_config=env_config,
        logger_config=logger_config
    )
    return agent_config

def main():

    parser = get_parser()
    args = parser.parse_args()

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=1)


if __name__ == '__main__':
    main()
