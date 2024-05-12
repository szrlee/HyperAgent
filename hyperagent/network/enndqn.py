from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from .common import BaseNet
import torch.nn.functional as F

ModuleType = Type[nn.Module]

class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        trainable: bool = True,
        bias_scale: float = 0.01,
        init_type: str = "xavier-uniform",
        use_bias: bool = True,
        device: Optional[Union[str, int, torch.device]] = "cpu",

    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.trainable = trainable
        self.bias_scale = bias_scale
        self.init_type = init_type
        self.use_bias = use_bias
        self.device = device

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        weight_init, bias_init = self.init_type.split("-")
        # init weight
        if weight_init == "zeros":
            nn.init.zeros_(self.weight)
        elif weight_init == "sphere":
            weight = np.random.randn(self.out_features * self.in_features).astype(np.float32)
            weight = weight / np.linalg.norm(weight)
            weight = weight.reshape(self.weight.shape)
            self.weight = nn.Parameter(torch.from_numpy(self.prior_std * weight).float())
        elif weight_init == "xavier":
            nn.init.xavier_normal_(self.weight)
        # init bias
        if self.use_bias:
            if bias_init == "zeros":
                nn.init.zeros_(self.bias)
            elif bias_init == "sphere":
                bias = np.random.randn(self.out_features).astype(np.float32)
                bias = bias / np.linalg.norm(bias)
                self.bias = nn.Parameter(torch.from_numpy(self.prior_std * bias).float())
            elif bias_init == "uniform":
                bound = 1 / np.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.use_bias:
            out = F.linear(x, self.weight, self.bias * self.bias_scale)
        else:
            out = F.linear(x, self.weight)
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        bias_scale: float = 0.01,
        init_type: str = "xavier-uniform",
        use_bias: bool = True,
        device: Optional[Union[str, int, torch.device]] = "cpu",
    ):
        super().__init__()
        linearlayer_params = dict(
            in_features=in_features, out_features=out_features,
            prior_std=prior_std, bias_scale=bias_scale,
            init_type=init_type, use_bias=use_bias, device=device,
        )
        self.based = LinearLayer(**linearlayer_params, trainable=True)
        if prior_std:
            self.prior = LinearLayer(**linearlayer_params, trainable=False)
            for param in self.prior.parameters():
                param.requires_grad = False

        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, x: torch.Tensor, prior_x: torch.Tensor = None, ) -> Tuple[torch.Tensor, Any]:
        out = self.based(x)
        if prior_x is not None and self.prior_std > 0:
            prior_out = self.prior(prior_x)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out


class EnsemblePrior(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        ensemble_num: int,
        ensemble_sizes: Sequence[int] = [5],
    ):
        super().__init__()
        self.basedmodel = nn.ModuleList([
            self.mlp(in_features, out_features, ensemble_sizes) for _ in range(ensemble_num)
        ])

        self.device = device
        self.ensemble_num = ensemble_num
        self.head_list = list(range(self.ensemble_num))

    def mlp(self, inp_dim, out_dim, hidden_sizes, bias=True):
        if len(hidden_sizes) == 0:
            return nn.Linear(inp_dim, out_dim, bias=bias)
        model = [nn.Linear(inp_dim, hidden_sizes[0], bias=bias)]
        model += [nn.ReLU(inplace=True)]
        for i in range(1, len(hidden_sizes)):
            model += [nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=bias)]
            model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(hidden_sizes[-1], out_dim, bias=bias)]
        return nn.Sequential(*model)

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, Any]:
        out = [self.basedmodel[k](x) for k in self.head_list]
        out = torch.stack(out, dim=1)
        out = torch.einsum("bza, bnz -> bna", out, noise)
        return out


class ENNLinear(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        state_dim: int,
        epinet_sizes: Sequence[int],
        noise_dim: int,
        noise_norm: int = 0,
        noise_scale: int = 0,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        use_bias: bool = False,
        epinet_init: str = "xavier_normal",
        device: Union[str, int, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.epinet_init = epinet_init
        self.in_features = noise_dim + hidden_dim + state_dim
        self.out_features = action_dim * noise_dim
        self.epinet = self._mlp(self.in_features, epinet_sizes, self.out_features, bias=use_bias)
        if prior_scale > 0:
            self.priornet = EnsemblePrior(state_dim, action_dim, device, noise_dim)
            for param in self.priornet.parameters():
                param.requires_grad = False
        self.reset_parameters()

        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.noise_norm = noise_norm
        self.noise_scale = noise_scale
        self.posterior_scale = posterior_scale
        self.prior_scale = prior_scale
        self.device = device

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'weight' in name:
                if self.epinet_init == "trunc_normal":
                    bound = 1.0 / np.sqrt(param.shape[-1])
                    torch.nn.init.trunc_normal_(param, std=bound, a=-2*bound, b=2*bound)
                elif self.epinet_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif self.epinet_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(param, gain=1.0)

    def _mlp(self, input_dim, hidden_sizes, output_dim, bias=True):
        layer_sizes = [input_dim] + hidden_sizes
        model = []
        for i in range(1, len(layer_sizes)):
            model += [nn.Linear(layer_sizes[i-1], layer_sizes[i], bias=bias)]
            model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(layer_sizes[-1], output_dim, bias=bias)]
        model = nn.Sequential(*model)
        return model

    def forward(self, x: torch.Tensor, feature: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        noise = self.process_noise(noise)
        batch_size = x.shape[0]
        hyper_inp = torch.cat([x, feature], dim=-1)
        hyper_inp = hyper_inp.unsqueeze(1).repeat(1, noise.shape[1], 1)
        hyper_inp = torch.cat([hyper_inp, noise], dim=-1)
        out = self.epinet(hyper_inp)
        out = out.view(batch_size, -1, self.action_dim, self.noise_dim)
        out = torch.einsum('bson, bsn -> bso', out, noise)
        if self.prior_scale > 0:
            prior_out = self.priornet(x, noise)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out

    def process_noise(self, noise):
        noise = noise.to(self.device)
        if self.noise_norm:
            noise /= torch.norm(noise, dim=-1, keepdim=True)
        elif self.noise_scale:
            noise /= np.sqrt(noise.shape[-1])
        return noise


class ENNNet(BaseNet):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        num_atoms: int = 1,
        base_prior: bool = True,
        feature_sg: bool = False,
        use_dueling: bool = False,
        model_type: str = 'mlp',
        normalize: str = '',
        embedding_dim: int = -1,
        last_layer_params: Dict[str, Any] = None
    ) -> None:
        last_layer = lambda x, y: ENNLinear(x, y, **last_layer_params)

        super().__init__(
            state_shape, action_shape, hidden_sizes, norm_layer, activation, device,
            softmax, num_atoms, use_dueling, base_prior, model_type, normalize, embedding_dim,
            last_layer=last_layer,
        )

        residual_params = {
            'device': device, 'prior_std': last_layer_params['prior_std'],
            'prior_scale': last_layer_params['prior_scale'],
            'posterior_scale': last_layer_params['posterior_scale'],
            'init_type': last_layer_params['bias_init'],
            'use_bias': False,
        }
        self.base_Q = Linear(
            self.feature.output_dim, num_atoms * int(np.prod(action_shape)), **residual_params
        )

        self.device = device
        self.feature_sg = feature_sg

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        noise: Dict[str, Any] = {},
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        if self.model_type == 'mlp':
            s = s.reshape(-1, self.input_dim)
        if isinstance(s, np.ndarray):
            s = s.astype(np.float32)
        elif isinstance(s, torch.Tensor):
            s = s.type(torch.float32)
        s = torch.as_tensor(s, device=self.device)
        hidden_logits, prior_hidden_logits = self.feature(s)
        bsz = hidden_logits.shape[0]
        base_q = self.base_Q(hidden_logits, prior_hidden_logits)
        base_q = base_q.view(bsz, 1, self.action_num, self.num_atoms)
        base_q = base_q.squeeze(dim=-1)
        if self.feature_sg:
            feautre = hidden_logits.detach()
        else:
            feautre = hidden_logits
        hyper_q = self.Q(s, feautre, noise=noise['Q'])
        hyper_q = hyper_q.view(bsz, -1, self.action_num, self.num_atoms)
        if self.use_dueling:
            hyper_v = hyper_v.view(bsz, -1, 1, self.num_atoms)
            hyper_q = hyper_q - hyper_q.mean(dim=-2, keepdim=True) + hyper_v
        hyper_q = hyper_q.squeeze(dim=-1)
        q = base_q + hyper_q
        logits = q
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state
