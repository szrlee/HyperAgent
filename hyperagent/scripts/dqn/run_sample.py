import argparse
import os
import time
import json
import pickle
import pprint
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from hyperagent.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from hyperagent.env import DummyVectorEnv
from hyperagent.env.utils import make_env
from hyperagent.policy import  DQNPolicy, C51Policy
from hyperagent.trainer import offpolicy_trainer
from hyperagent.utils import TensorboardLogger, import_module_or_data, read_config_dict
from hyperagent.network.linear_network import LinearNet


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument('--task', type=str, default='DeepSea-v0')
    parser.add_argument('--max-step', type=int, default=500)
    parser.add_argument('--size', type=int, default=20, help="only for DeepSea-v0")
    parser.add_argument('--length', type=int, default=50, help="only for MDP-v1/v2")
    parser.add_argument('--final-reward', type=int, default=1, help="only for MDP-v1. If it is not 0 or 1, it means randomly generated")
    parser.add_argument('--other-reward', type=float, default=0.0, help="only for MDP-v1/DeepSea-v0")
    parser.add_argument('--move-cost', type=float, default=0.01, help="only for MDP-v1/DeepSea-v0")
    parser.add_argument('--random-dynamics-scale', type=float, default=0.)
    parser.add_argument('--random-reward-scale', type=float, default=0.)
    parser.add_argument('--padding-obs', type=int, default=0, choices=[0, 1], help="only for MDP-v1/v2")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--norm-obs', action="store_true", default=False)
    parser.add_argument('--norm-ret', action="store_true", default=False)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    # training config
    parser.add_argument('--target-update-freq', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-eps', type=float, default=0.00015)
    parser.add_argument('--loss-scale', type=float, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--v-max', type=float, default=0.)
    parser.add_argument('--num-atoms', type=int, default=1)
    parser.add_argument('--num-quantiles', type=int, default=1)
    parser.add_argument('--clip-value', type=int, default=0, choices=[0, 1])
    parser.add_argument('--clip-grad-norm', type=float, default=10.)
    parser.add_argument('--grad-coef', type=float, default=0.0)
    # algorithm config
    parser.add_argument('--alg-type', type=str, default="linear")
    parser.add_argument('--prior-std', type=float, default=0, help="Greater than 0 means using priormodel")
    parser.add_argument('--prior-scale', type=float, default=10)
    parser.add_argument('--posterior-scale', type=float, default=1)
    # network config
    parser.add_argument('--hidden-layer', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--use-dueling', type=int, default=0, choices=[0, 1])
    parser.add_argument('--is-double', type=int, default=1, choices=[0, 1])
    parser.add_argument('--base-prior', type=int, default=0, choices=[0, 1])
    parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'conv', 'conv_v2'])
    parser.add_argument('--init-type', type=str, default="xavier_normal", choices=["", "trunc_normal", "xavier_uniform", "xavier_normal"])
    # epoch config
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--episode-per-test', type=int, default=100)
    # buffer confing
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--min-buffer-size', type=int, default=128)
    parser.add_argument('--prioritized', type=int, default=0, choices=[0, 1])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    # action selection confing
    parser.add_argument('--random-start', type=int, default=1, choices=[0, 1], help="1: use random policy to collect minibuffersize data")
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    # other confing
    parser.add_argument('--save-interval', type=int, default=4)
    parser.add_argument('--save-buffer', action="store_true", default=False)
    parser.add_argument('--logdir', type=str, default='~/results/hyperagent/deepsea')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # overwrite config
    parser.add_argument('--config', type=str, default="{}",
                        help="game config eg., {}")
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # environment
    if args.task.startswith("DeepSea"):
        env_config = {
            'seed':args.seed, 'size': args.size, 'mapping_seed': args.seed,
            'unscaled_move_cost': args.move_cost, 'other_reward': args.other_reward,
            'random_dynamics_scale': args.random_dynamics_scale, 'random_reward_scale': args.random_reward_scale,
        }
    elif args.task.startswith("MDP"):
        env_config = {
            'seed':args.seed, 'length': args.length, 'move_cost': args.move_cost,
            'final_reward': args.final_reward, 'other_reward': args.other_reward, 'padding_obs': args.padding_obs,
        }
    elif args.task.startswith("Radar"):
        env_config = {'episode': args.max_step}
    else:
        env_config = {}
    def make_thunk():
        return lambda: make_env(env_name=args.task, max_step=args.max_step, env_config=env_config)
    # you can also use hyperagent.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([make_thunk()], norm_obs=args.norm_obs)
    test_envs = DummyVectorEnv([make_thunk()], norm_obs=args.norm_obs)
    args.value_range = ()
    if args.task.startswith('DeepSea') or args.task.startswith('MDP'):
        train_action_mappling = np.array([action_mapping() for action_mapping in train_envs._get_action_mapping])
        test_action_mappling = np.array([action_mapping() for action_mapping in test_envs._get_action_mapping])
        assert (train_action_mappling == test_action_mappling).all()
        args.max_step = args.size
        args.value_range = (-2, 2) if args.clip_value else ()
    if args.task.startswith('MDP'):
        train_all_rewards = np.array([get_rewards() for get_rewards in train_envs._get_rewards])
        test_all_rewards = np.array([get_rewards() for get_rewards in test_envs._get_rewards])
        assert (train_all_rewards == test_all_rewards).all()
        args.max_step = args.length
        args.final_reward = train_all_rewards[0][0]
    args.state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
    args.action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed)

    # model
    if args.num_atoms > 1:
        assert args.num_quantiles == 1
    if args.num_quantiles > 1:
        assert args.num_atoms == 1
    last_layer_params = {
        'device': args.device,
        'prior_std': args.prior_std,
        'prior_scale': args.prior_scale,
        'posterior_scale': args.posterior_scale,
    }
    args.hidden_sizes = [args.hidden_size] * args.hidden_layer
    args.softmax = True if args.num_atoms > 1 else False
    model_params = {
        "state_shape": args.state_shape,
        "action_shape": args.action_shape,
        "hidden_sizes": args.hidden_sizes,
        "device": args.device,
        "softmax": args.softmax,
        "num_atoms": args.num_atoms * args.num_quantiles,
        "use_dueling": args.use_dueling,
        "base_prior": args.base_prior,
        "model_type": args.model_type,
        "last_layer_params": last_layer_params
    }
    model = LinearNet(**model_params).to(args.device)

    # init model
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param)
        else:
            bound = 1.0 / np.sqrt(param.shape[-1])
            torch.nn.init.trunc_normal_(param, std=bound, a=-2*bound, b=2*bound)

    param_dict = {"Non-trainable": [], "Trainable": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param_dict["Non-trainable"].append(name)
        else:
            param_dict["Trainable"].append(name)
    pprint.pprint(param_dict)
    print(f"Network structure:\n{str(model)}")
    print(f"Network parameters: {sum(param.numel() for param in model.parameters())}")

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.lr_eps)

    # policy
    policy_params = {
        "model": model,
        "optim": optim,
        "discount_factor": args.gamma,
        "estimation_step": args.n_step,
        "target_update_freq": args.target_update_freq,
        "reward_normalization": args.norm_ret,
        "is_double": args.is_double,
        "loss_scale": args.loss_scale,
        "clip_grad_norm": args.clip_grad_norm,
        "grad_coef": args.grad_coef,
    }
    if args.num_atoms > 1:
        policy_params.update({'num_atoms': args.num_atoms, 'v_max': args.v_max, 'v_min': -args.v_max})
        policy = C51Policy(**policy_params).to(args.device)
    else:
        policy = DQNPolicy(**policy_params).to(args.device)

    # buffer
    if args.prioritized:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size, buffer_num=len(train_envs),
            alpha=args.alpha, beta=args.beta, weight_norm=True
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    game_name = args.task[:-3].lower()
    if args.task.startswith('DeepSea'):
        game_name += f'{args.size}'
    elif args.task.startswith('MDP'):
        game_name += f'{args.task[-2:]}_{args.length}'
    log_file = f"{args.alg_type}_{game_name}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.join(args.logdir, args.task, log_file)
    log_path = os.path.expanduser(log_path)
    os.makedirs(os.path.expanduser(log_path), exist_ok=True)
    logger = TensorboardLogger(log_path, save_interval=args.save_interval)
    with open(os.path.join(log_path, "config.json"), "wt") as f:
        kvs = vars(args)
        f.write(json.dumps(kvs, indent=4) + '\n')
        f.flush()
        f.close()

    def train_fn(epoch, env_step, env_episode):
        policy.set_eps(args.eps_train)

    def test_fn(epoch, env_step, env_episode):
        policy.set_eps(args.eps_test)

    # trainer
    # args.step_per_collect *= args.training_num
    args.update_per_step /= args.step_per_collect
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        learning_start=args.min_buffer_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        logger=logger,
        save_checkpoint_fn=None,
        verbose=True,
    )
    # assert stop_fn(result['best_reward'])
    # pprint.pprint(result)


if __name__ == '__main__':
    args = get_args()
    config = read_config_dict(args.config)
    for k, v in config.items():
        if k not in args.__dict__.keys():
            print(f'unrecognized config k: {k}, v: {v}, ignored')
            continue
        args.__dict__[k] = v
    main(args)
