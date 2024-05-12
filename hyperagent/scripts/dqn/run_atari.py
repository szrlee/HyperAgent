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
from hyperagent.policy import  DQNPolicy
from hyperagent.trainer import offpolicy_trainer
from hyperagent.utils import TensorboardLogger
from hyperagent.network import DQNNet

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--atari-100k', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--norm-ret', action="store_true", default=False)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    # training config
    parser.add_argument('--target-update-freq', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-eps', type=float, default=0.00015)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip-grad-norm', type=float, default=10.)
    parser.add_argument('--grad-coef', type=float, default=0.01)
    # algorithm config
    parser.add_argument('--alg-type', type=str, default="DQN")
    parser.add_argument('--prior-std', type=float, default=0)
    parser.add_argument('--prior-scale', type=float, default=0.1)
    parser.add_argument('--posterior-scale', type=float, default=0.1)
    # network config
    parser.add_argument('--normalize', type=str, default='MinMax', choices=['', 'MinMax', 'Layer'])
    parser.add_argument('--hidden-layer', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--is-double', type=int, default=1, choices=[0, 1])
    parser.add_argument('--base-prior', type=int, default=0, choices=[0, 1])
    parser.add_argument('--model-type', type=str, default='conv', choices=['mlp', 'conv', 'conv_v2'])
    parser.add_argument('--init-type', type=str, default="xavier_normal", choices=["", "trunc_normal", "xavier_uniform", "xavier_normal"])
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
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train-start', type=float, default=1)
    parser.add_argument('--eps-train-end', type=float, default=0.1)
    # other confing
    parser.add_argument('--save-interval', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='~/results/hyperagent/atari')
    parser.add_argument('--logfile', type=str, default=None)
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
        'prior_std': args.prior_std,
        'prior_scale': args.prior_scale,
        'posterior_scale': args.posterior_scale,
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
    model = DQNNet(**model_params).to(args.device)

    # init model
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param)
        elif 'weight' in name:
            if args.init_type == "trunc_normal":
                bound = 1.0 / np.sqrt(param.shape[-1])
                torch.nn.init.trunc_normal_(param, std=bound, a=-2*bound, b=2*bound)
            elif args.init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(param, gain=1.0)
            elif args.init_type == "xavier_normal":
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
        "clip_grad_norm": args.clip_grad_norm,
        "grad_coef": args.grad_coef,
    }
    policy = DQNPolicy(**policy_params).to(args.device)

    # buffer
    buf = VectorReplayBuffer(
        args.buffer_size, buffer_num=len(train_envs),
        ignore_obs_next=True, save_only_last_obs=True, stack_num=args.frames_stack
    )

    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    if args.logfile is None:
        game_name = args.task[:args.task.find('No')].lower()
        args.logfile = f"{args.alg_type}_{game_name}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.expanduser(os.path.join(args.logdir, args.task, args.logfile))
    os.makedirs(os.path.expanduser(log_path), exist_ok=True)
    logger = TensorboardLogger(log_path, save_interval=args.save_interval)
    with open(os.path.join(log_path, "config.json"), "wt") as f:
        kvs = vars(args)
        f.write(json.dumps(kvs, indent=4) + '\n')
        f.flush()
        f.close()

    def save_fn(policy, env_step=None):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step, env_episode):
        if int(args.epoch * args.step_per_epoch) == int(1e5):
            duration = int(40e3)
        elif int(args.epoch * args.step_per_epoch) == int(5e5):
            duration = int(200e3)
        elif int(args.epoch * args.step_per_epoch) == int(2e6):
            duration = int(200e3)
        eps = linear_schedule(
            start_e=args.eps_train_start, end_e=args.eps_train_end, duration=duration, t=env_step
        )
        policy.set_eps(eps)

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
        random_start=args.random_start,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        save_fn=save_fn,
        logger=logger,
    )
    # assert stop_fn(result['best_reward'])
    # pprint.pprint(result)
    if args.evaluation:
        eval_envs = DummyVectorEnv(
            [lambda: make_atari_env_watch(args) for _ in range(20)]
        )
        # reset seed
        eval_envs.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(args.seed)
        test(log_path, eval_envs, policy, args.eps_test)

def test(log_path, env, policy: torch.nn.Module, eps_test: float):
    from hyperagent.utils.logger.tensorboard import CSVOutputFormat

    policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))
    if eps_test > 0:
        policy.set_eps(eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    best_results = collector.collect(n_episode=200)

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