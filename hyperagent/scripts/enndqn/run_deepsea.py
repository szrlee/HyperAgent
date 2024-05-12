import argparse
import os
import time
import json
import pprint
import numpy as np
import torch

from hyperagent.data import Collector, ReplayBuffer
from hyperagent.env import DummyVectorEnv
from hyperagent.env.utils import make_env
from hyperagent.trainer import offpolicy_trainer
from hyperagent.utils import TensorboardLogger, import_module_or_data, read_config_dict
from hyperagent.network import ENNNet
from hyperagent.policy import HyperAgentPolicy


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument('--task', type=str, default='DeepSea-v0')
    parser.add_argument('--size', type=int, default=20, help="size of DeepSea-v0")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--norm-ret', action="store_true", default=False)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    # training config
    parser.add_argument('--max-target', type=float, default=0.0)
    parser.add_argument('--target-noise-per-sample', type=int, default=1)
    parser.add_argument('--noise-per-sample', type=int, default=20)
    parser.add_argument('--target-update-freq', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-eps', type=float, default=0.00015)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip-grad-norm', type=float, default=10.)
    # algorithm config
    parser.add_argument('--alg-type', type=str, default="ENNDQN")
    parser.add_argument('--train-iter', type=int, default=0)
    parser.add_argument('--noise-norm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--noise-scale', type=int, default=0, choices=[0, 1])
    parser.add_argument('--noise-std', type=float, default=1.)
    parser.add_argument('--noise-dim', type=int, default=4)
    parser.add_argument('--prior-std', type=float, default=1.0)
    parser.add_argument('--prior-scale', type=float, default=1.)
    parser.add_argument('--posterior-scale', type=float, default=1.)
    parser.add_argument('--target-noise-coef', type=float, default=0.)
    parser.add_argument('--target-noise-type', type=str, default="sp", choices=["ssp", "sp", "gs", "bi", "exp", "a-exp", "uni"])
    parser.add_argument('--hyper-reg-coef', type=float, default=0.)
    parser.add_argument('--grad-coef', type=float, default=0.0)
    parser.add_argument('--dependent-target-noise', type=int, default=0, choices=[0, 1])
    parser.add_argument('--state-dependent-noise-action', type=int, default=0, choices=[0, 1])
    parser.add_argument('--state-dependent-noise-target', type=int, default=1, choices=[0, 1])
    # network config
    parser.add_argument('--normalize', type=str, default='', choices=['', 'MinMax', 'Layer'])
    parser.add_argument('--epinet-layer', type=int, default=1)
    parser.add_argument('--epinet-size', type=int, default=50)
    parser.add_argument('--residual-link', type=int, default=1, choices=[0, 1])
    parser.add_argument('--residual-scale', type=float, default=1.0)
    parser.add_argument('--feature-sg', type=int, default=1, choices=[0, 1])
    parser.add_argument('--feature-version', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--hidden-layer', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--embedding-dim', type=int, default=-1)
    parser.add_argument('--use-dueling', type=int, default=0, choices=[0, 1])
    parser.add_argument('--base-dueling', type=int, default=0, choices=[0, 1])
    parser.add_argument('--is-double', type=int, default=1, choices=[0, 1])
    parser.add_argument('--base-prior', type=int, default=0, choices=[0, 1])
    parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'conv', 'conv_v2'])
    parser.add_argument('--base-init', type=str, default="xavier_normal", choices=["", "trunc_normal", "xavier_uniform", "xavier_normal"])
    parser.add_argument('--epinet-init', type=str, default="xavier_normal", choices=["", "trunc_normal", "xavier_uniform", "xavier_normal"])
    parser.add_argument('--bias-init', type=str, default="xavier-uniform",
        choices=["default", "zeros-zeros", "sphere-sphere", "xavier-uniform", "sphere-pos", "sphere-neg", "xavier-pos", "xavier-neg"]
    )
    # epoch config
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--episode-per-test', type=int, default=100)
    # buffer confing
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--min-buffer-size', type=int, default=128)
    # action selection confing
    parser.add_argument('--random-start', type=int, default=1, choices=[0, 1], help="1: use random policy to collect minibuffersize data")
    parser.add_argument('--action-sample-num', type=int, default=1)
    parser.add_argument('--action-select-scheme', type=str, default="Greedy", choices=['Greedy', 'MAX'])
    parser.add_argument('--quantile-max', type=float, default=0.8)
    # other confing
    parser.add_argument('--save-interval', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='~/results/hyperagent/deepsea')
    parser.add_argument('--logfile', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # environment
    env_config = {'seed':args.seed, 'size': args.size, 'mapping_seed': args.seed}
    def make_thunk():
        return lambda: make_env(env_name=args.task, env_config=env_config)
    train_envs = DummyVectorEnv([make_thunk() for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([make_thunk() for _ in range(args.test_num)])

    train_action_mappling = np.array([action_mapping() for action_mapping in train_envs._get_action_mapping])
    test_action_mappling = np.array([action_mapping() for action_mapping in test_envs._get_action_mapping])
    assert (train_action_mappling == test_action_mappling).all()

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
    last_layer_params = {
        'state_dim': np.prod(args.state_shape),
        'epinet_sizes': [args.epinet_size] * args.epinet_layer,
        'noise_dim': args.noise_dim,
        'noise_norm': args.noise_norm,
        'noise_scale': args.noise_scale,
        'prior_std': args.prior_std,
        'prior_scale': args.prior_scale,
        'posterior_scale': args.posterior_scale,
        'epinet_init': args.epinet_init,
        'bias_init': args.bias_init,
        'use_bias': False,
        'device': args.device,
    }
    args.hidden_sizes = [args.hidden_size] * args.hidden_layer
    model_params = {
        "state_shape": args.state_shape,
        "action_shape": args.action_shape,
        "hidden_sizes": args.hidden_sizes,
        "device": args.device,
        "base_prior": args.base_prior,
        "feature_sg": args.feature_sg,
        "model_type": args.model_type,
        "normalize": args.normalize,
        "last_layer_params": last_layer_params
    }
    model = ENNNet(**model_params).to(args.device)

    # init base model
    for name, param in model.named_parameters():
        if name.startswith('feature') and "embedding" not in name:
            if 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'weight' in name:
                if args.base_init == "trunc_normal":
                    bound = 1.0 / np.sqrt(param.shape[-1])
                    torch.nn.init.trunc_normal_(param, std=bound, a=-2*bound, b=2*bound)
                elif args.base_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif args.base_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(param, gain=1.0)

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
    trainable_params = [{'params': (p for name, p in model.named_parameters() if 'prior' not in name)}]
    optim = torch.optim.Adam(trainable_params, lr=args.lr, eps=args.lr_eps)

    # policy
    policy_params = {
        "model": model,
        "optim": optim,
        "discount_factor": args.gamma,
        "estimation_step": args.n_step,
        "target_update_freq": args.target_update_freq,
        "reward_normalization": args.norm_ret,
        "is_double": args.is_double,
        "noise_per_sample": args.noise_per_sample,
        "target_noise_per_sample": args.target_noise_per_sample,
        "max_target": args.max_target,
        "action_sample_num": args.action_sample_num,
        "action_select_scheme": args.action_select_scheme,
        "quantile_max": args.quantile_max,
        "noise_std": args.noise_std,
        "noise_dim": args.noise_dim,
        "grad_coef": args.grad_coef,
        "hyper_reg_coef": args.hyper_reg_coef,
        "target_noise_coef": args.target_noise_coef,
        "target_noise_type": args.target_noise_type,
        "clip_grad_norm": args.clip_grad_norm,
        "dependent_target_noise": args.dependent_target_noise,
        "state_dependent_noise_action": args.state_dependent_noise_action,
        "state_dependent_noise_target": args.state_dependent_noise_target,
        "seed": args.seed,
        "action_space": train_envs.observation_space[0],
    }
    policy = HyperAgentPolicy(**policy_params).to(args.device)

    # buffer
    buf = ReplayBuffer(args.buffer_size)

    # collector
    args.target_noise_dim = args.noise_dim
    train_collector = Collector(
        policy, train_envs, buf, exploration_noise=False,
        target_noise_dim=args.target_noise_dim, target_noise_type=args.target_noise_type,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # log
    if args.logfile is None:
        game_name = args.task[:-3].lower() + f'{args.size}'
        args.logfile = f"{args.alg_type}_{args.action_select_scheme}_{game_name}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.join(args.logdir, args.task, args.logfile)
    log_path = os.path.expanduser(log_path)
    os.makedirs(os.path.expanduser(log_path), exist_ok=True)
    logger = TensorboardLogger(log_path, save_interval=args.save_interval)
    with open(os.path.join(log_path, "config.json"), "wt") as f:
        kvs = vars(args)
        f.write(json.dumps(kvs, indent=4) + '\n')
        f.flush()
        f.close()

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
        random_start=args.random_start,
        update_per_step=args.update_per_step,
        logger=logger,
        verbose=True,
    )
    # assert stop_fn(result['best_reward'])
    pprint.pprint(result)


if __name__ == '__main__':
    main(get_args())
