import numpy as np
import torch
import torch.nn.functional as F
from hyperagent.data import to_numpy, to_torch_as

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)

class BootstrappedNoise():
    def __init__(self, noise_type='gs', noise_coef=0.01) -> None:
        self.noise_type = noise_type
        self.noise_coef = noise_coef

    def compute_noise(self, update_noise, target_noise):
        target_noise = to_torch_as(target_noise, update_noise)
        if len(target_noise.shape) < len(update_noise.shape):
            target_noise = target_noise.unsqueeze(1)
        if target_noise.shape[1] != update_noise.shape[1]:
            target_noise = target_noise.repeat(1, update_noise.shape[1], 1)
        if self.noise_type in ['ssp', 'sp', 'gs', 'uni', "bi"]:
            loss_noise = (update_noise * target_noise).sum(-1)
        elif self.noise_type == 'gbi':
            loss_noise = (update_noise * target_noise).sum(-1)
            loss_noise = torch.sign(loss_noise) + 1
        elif self.noise_type in ['a-exp', 'exp']:
            noise_dim = update_noise.shape[-1]
            u_noise_1, u_noise_2 = update_noise[:, :, :noise_dim // 2], update_noise[:, :, noise_dim // 2:]
            t_noise_1, t_noise_2 = target_noise[:, :, :noise_dim // 2], target_noise[:, :, noise_dim // 2:]
            loss_noise = 0.5 * ((u_noise_1 * t_noise_1).sum(-1).square() + (u_noise_2 * t_noise_2).sum(-1).square())
        loss_noise *= self.noise_coef
        return loss_noise

class SampleNoise():
    def __init__(
        self,
        noise_dim: int,
        noise_std: float = 1.,
        one_hot_noise: bool = False,
        seed: int = 2022,
    ):
        self.noise_std = noise_std
        self.noise_dim = noise_dim
        self.one_hot_noise = one_hot_noise

        self.sample_test_noise = self._sample_test_noise
        self.sample_train_noise = self._sample_train_noise
        self.sample_update_noise = self._sample_update_noise
        self.sample_target_noise = self._sample_target_noise
        self.noise_rng = {}
        for i, name in enumerate(["train", "test", "update", "target"]):
            rng = torch.Generator()
            rng.manual_seed(seed + i)
            self.noise_rng[name] = rng

    def reset_seed(self, seed: int, rng_name: str):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.noise_rng[rng_name] = rng

    def gen_noise(self, noise_num: tuple, rng_name: str):
        noise_shape = noise_num + (self.noise_dim, )
        noise = torch.randn(size=noise_shape, generator=self.noise_rng[rng_name]) * self.noise_std
        return noise

    def gen_one_hot_noise(self, noise_num: tuple, rng_name: str):
        noise = torch.randint(0, self.noise_dim, noise_num, generator=self.noise_rng[rng_name])
        noise = F.one_hot(noise, self.noise_dim).to(torch.float32)
        return noise

    def _sample_target_noise(self, noise_num: tuple):
        if self.one_hot_noise:
            hyper_noise = self.gen_one_hot_noise(noise_num, "target")
        else:
            hyper_noise = self.gen_noise(noise_num, "target")
        noise = {'Q': hyper_noise}
        return noise

    def _sample_update_noise(self, noise_num: tuple):
        if self.one_hot_noise:
            batch_size = noise_num[0]
            hyper_noise = torch.arange(self.noise_dim).unsqueeze(0).repeat(batch_size, 1)
            hyper_noise = F.one_hot(hyper_noise, self.noise_dim).to(torch.float32)
        else:
            hyper_noise = self.gen_noise(noise_num, "update")
        noise = {'Q': hyper_noise}
        return noise

    def _sample_train_noise(self, noise_num: tuple):
        if self.one_hot_noise:
            hyper_noise = self.gen_one_hot_noise(noise_num, "train")
        else:
            hyper_noise = self.gen_noise(noise_num, "train")
        noise = {'Q': hyper_noise}
        return noise

    def _sample_test_noise(self, noise_num: tuple):
        if self.one_hot_noise:
            hyper_noise = self.gen_one_hot_noise(noise_num, "test")
        else:
            hyper_noise = self.gen_noise(noise_num, "test")
        noise = {'Q': hyper_noise}
        return noise

class ActionSelection():
    def __init__(self, select_scheme, quantile_max=1.0):
        self.select_scheme = select_scheme
        self.quantile_max = quantile_max
        if select_scheme == "Greedy":
            self.get_actions = getattr(self, '_greedy_action_select')
        elif select_scheme == 'MAX':
            self.get_actions = getattr(self, '_max_action_select')
        else:
            raise NotImplementedError(f'No action selcet scheme {select_scheme}')

    def _max_action_select(self, q):
        q = torch.quantile(q, self.quantile_max, dim=1)
        q = to_numpy(q)
        act = [rd_argmax(q[i]) for i in range(q.shape[0])]
        act = np.array(act)
        return act

    def _greedy_action_select(self, q):
        q = to_numpy(q.squeeze(1))
        act = [rd_argmax(q[i]) for i in range(q.shape[0])]
        act = np.array(act)
        return act
