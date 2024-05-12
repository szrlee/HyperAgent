import argparse
import os
import time
import json
import pprint
import numpy as np
import torch

from hyperagent.data import Collector, VectorReplayBuffer
from hyperagent.env import DummyVectorEnv
from hyperagent.env.utils import make_atari_env, make_atari_env_watch
from hyperagent.trainer import offpolicy_trainer
from hyperagent.utils import TensorboardLogger
from hyperagent.network import HyperAgentNet
from hyperagent.policy import HyperAgentPolicy


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--atari-100k', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--norm-ret', action="store_true", default=False)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    # training config
    parser.add_argument('--max-target', type=float, default=0.0)
    parser.add_argument('--target-noise-per-sample', type=int, default=1)
    parser.add_argument('--noise-per-sample', type=int, default=20)
    parser.add_argument('--target-update-freq', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-eps', type=float, default=0.00015)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip-grad-norm', type=float, default=10.)
    # algorithm config
    parser.add_argument('--alg-type', type=str, default="HyperAgent")
    parser.add_argument('--noise-norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--noise-scale', type=int, default=0, choices=[0, 1])
    parser.add_argument('--noise-std', type=float, default=1.)
    parser.add_argument('--noise-dim', type=int, default=4)
    parser.add_argument('--prior-std', type=float, default=0.1)
    parser.add_argument('--prior-scale', type=float, default=0.1)
    parser.add_argument('--posterior-scale', type=float, default=0.1)
    parser.add_argument('--target-noise-coef', type=float, default=0.01)
    parser.add_argument('--target-noise-type', type=str, default="sp", choices=["ssp", "sp", "gs", "gbi", "bi", "exp", "a-exp", "uni"])
    parser.add_argument('--hyper-reg-coef', type=float, default=0.01)
    parser.add_argument('--grad-coef', type=float, default=0.01)
    parser.add_argument('--dependent-target-noise', type=int, default=0, choices=[0, 1])
    parser.add_argument('--state-dependent-noise-action', type=int, default=0, choices=[0, 1])
    parser.add_argument('--state-dependent-noise-target', type=int, default=1, choices=[0, 1])
    # network config
    parser.add_argument('--normalize', type=str, default='MinMax', choices=['', 'MinMax', 'Layer'])
    parser.add_argument('--hidden-layer', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--is-double', type=int, default=1, choices=[0, 1])
    parser.add_argument('--base-prior', type=int, default=1, choices=[0, 1])
    parser.add_argument('--model-type', type=str, default='conv', choices=['mlp', 'conv', 'conv_v2'])
    parser.add_argument('--base-init', type=str, default="xavier_normal", choices=["", "trunc_normal", "xavier_uniform", "xavier_normal"])
    parser.add_argument('--hyper-init', type=str, default="xavier_normal", choices=["", "trunc_normal", "xavier_uniform", "xavier_normal"])
    parser.add_argument('--prior-init', type=str, default="sDB", choices=["", "sDB", "gDB", "trunc_normal", "xavier_uniform", "xavier_normal"])
    parser.add_argument('--bias-init', type=str, default="xavier-uniform",
        choices=["default", "zeros-zeros", "sphere-sphere", "xavier-uniform", "sphere-pos", "sphere-neg", "xavier-pos", "xavier-neg"]
    )
    # epoch config
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--episode-per-test', type=int, default=10)
    # buffer confing
    parser.add_argument('--buffer-size', type=int, default=int(1e5))
    parser.add_argument('--min-buffer-size', type=int, default=2000)
    # action selection confing
    parser.add_argument('--random-start', type=int, default=1, choices=[0, 1], help="1: use random policy to collect minibuffersize data")
    parser.add_argument('--action-sample-num', type=int, default=1)
    parser.add_argument('--action-select-scheme', type=str, default="Greedy", choices=['Greedy', 'MAX'])
    parser.add_argument('--quantile-max', type=float, default=0.8)
    # other confing
    parser.add_argument('--save-interval', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='~/results/hyperagent/atari')
    parser.add_argument('--logfile', type=str, default=None)
    parser.add_argument('--evaluation', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # environment
    train_envs = DummyVectorEnv([lambda: make_atari_env(args) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: make_atari_env_watch(args) for _ in range(args.test_num)])
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
        'device': args.device,
        'noise_dim': args.noise_dim,
        'noise_norm': args.noise_norm,
        'noise_scale': args.noise_scale,
        'prior_std': args.prior_std,
        'prior_scale': args.prior_scale,
        'posterior_scale': args.posterior_scale,
        'hyper_init': args.hyper_init,
        'prior_init': args.prior_init,
        'bias_init': args.bias_init,
    }
    args.hidden_sizes = [args.hidden_size] * args.hidden_layer
    model_params = {
        "state_shape": args.state_shape,
        "action_shape": args.action_shape,
        "hidden_sizes": args.hidden_sizes,
        "device": args.device,
        "base_prior": args.base_prior,
        "model_type": args.model_type,
        "normalize": args.normalize,
        "last_layer_params": last_layer_params
    }
    model = HyperAgentNet(**model_params).to(args.device)

    # init base model
    for name, param in model.named_parameters():
        if name.startswith('feature'):
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
    buf = VectorReplayBuffer(
        args.buffer_size, buffer_num=len(train_envs),
        ignore_obs_next=True, save_only_last_obs=True, stack_num=args.frames_stack
    )

    # collector
    args.target_noise_dim = args.noise_dim
    train_collector = Collector(
        policy, train_envs, buf, exploration_noise=False,
        target_noise_dim=args.target_noise_dim, target_noise_type=args.target_noise_type,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # log
    if args.logfile is None:
        game_name = args.task[:args.task.find('No')].lower()
        args.logfile = f"{args.alg_type}_{args.action_select_scheme}_{game_name}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.expanduser(os.path.join(args.logdir, args.task, args.logfile))
    os.makedirs(os.path.expanduser(log_path), exist_ok=True)
    logger = TensorboardLogger(log_path, save_interval=args.save_interval)
    with open(os.path.join(log_path, "config.json"), "wt") as f:
        kvs = vars(args)
        f.write(json.dumps(kvs, indent=4) + '\n')
        f.flush()
        f.close()

    def save_fn(policy, env_step=None):
        if env_step is None:
            policy_name = "policy"
        else:
            if env_step <= 100e3:
                policy_name = "policy_100K"
            elif env_step <= 500e3:
                policy_name = "policy_500K"
            elif env_step <= 1e6:
                policy_name = "policy_1M"
            elif env_step <= 2e6:
                policy_name = "policy_2M"
            elif env_step <= 5e6:
                policy_name = "policy_5M"
            elif env_step <= 20e6:
                policy_name = "policy_20M"
            else:
                policy_name = "policy"
        torch.save(policy.state_dict(), os.path.join(log_path, f'{policy_name}.pth'))
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # trainer
    args.step_per_collect *= args.training_num
    args.update_per_step *= args.training_num
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
        save_fn=save_fn,
        logger=logger,
        verbose=True,
    )
    # assert stop_fn(result['best_reward'])
    # pprint.pprint(result)
    if args.evaluation > 0:
        eval_envs = DummyVectorEnv(
            [lambda: make_atari_env_watch(args) for _ in range(args.evaluation // 10)]
        )
        # reset seed
        policy.reset_test_noise(args.seed)
        eval_envs.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(args.seed)
        test(log_path, eval_envs, policy, args.evaluation)


def test(log_path, env, policy: torch.nn.Module, n_episode: int):
    from hyperagent.utils.logger.tensorboard import CSVOutputFormat

    policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))
    collector = Collector(policy, env)
    best_results = collector.collect(n_episode=n_episode)

    results = {
        "best/rew": best_results["rew"],
        "best/rew_std": best_results["rew_std"],
    }
    with open(os.path.join(log_path, "results.json"), "wt") as f:
        f.write(json.dumps(results, indent=4) + '\n')
        f.flush()
        f.close()

    logger = CSVOutputFormat(os.path.join(log_path, 'results.csv'))
    for best_rew in best_results['rews']:
        logger.writekvs({"best/rew": best_rew})


if __name__ == '__main__':
    main(get_args())
