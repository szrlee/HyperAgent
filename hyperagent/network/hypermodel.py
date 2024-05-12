from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperagent import HyperLayer, HyperLinear

ModuleType = Type[nn.Module]



class HyperLinearV2(HyperLinear):
    def base_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor=None):
        weight = weight.reshape((weight.shape[0], -1,) + self.weight_shape) # (batch, noise_num, in_dim, out_dim)
        if weight.shape[0] == x.shape[0]:
            out = torch.einsum('bnh, bnha -> bna', x, weight)
        else:
            out = torch.einsum('bh, nha -> bna', x, weight.squeeze(0))
        if bias is not None:
            bias = bias.reshape((weight.shape[0], -1,) + self.bias_shape) * self.bias_scale # (batch, noise_num, out_dim)
            out += bias
        return out

    def forward(self, x: torch.Tensor, prior_x: torch.Tensor = None, noise: torch.Tensor = None) -> torch.Tensor:
        hyper_noise = self.process_noise(noise)
        weight = self.hyper_weight(hyper_noise)
        out = self.base_forward(x, weight)
        prior_out = None
        if prior_x is not None and self.prior_std > 0:
            prior_weight = self.prior_weight(hyper_noise)
            prior_out = self.base_forward(prior_x, prior_weight)
        return out, prior_out



class HyperModel(nn.Module):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        num_atoms: int = 1,
        last_layer_params: Dict[str, Any] = None
    ) -> None:
        super().__init__()
        input_dim = int(np.prod(state_shape))
        output_dim = int(np.prod(action_shape))
        dims = [input_dim] + hidden_sizes + [output_dim]

        last_layer = lambda x, y: HyperLinearV2(x, y, **last_layer_params)
        self.layer1 = last_layer(dims[0], dims[1])
        self.layer2 = last_layer(dims[1], dims[2])
        self.layer3 = last_layer(dims[2], dims[3])

        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_num = int(np.prod(action_shape))
        self.num_atoms = num_atoms

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        noise: Dict[str, Any] = {},
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        if isinstance(s, np.ndarray):
            s = s.astype(np.float32)
        elif isinstance(s, torch.Tensor):
            s = s.type(torch.float32)
        s = torch.as_tensor(s, device=self.device)
        s = s.reshape(-1, 1, self.input_dim)
        hs, prior_hs = s, s
        hs, prior_hs = self.layer1(hs, prior_hs, noise['Q'])
        hs, prior_hs = F.relu(hs), F.relu(prior_hs)
        hs, prior_hs = self.layer2(hs, prior_hs, noise['Q'])
        hs, prior_hs = F.relu(hs), F.relu(prior_hs)
        hs, prior_hs = self.layer3(hs, prior_hs, noise['Q'])
        bsz = hs.shape[0]
        q = hs * self.layer3.posterior_scale + prior_hs * self.layer3.prior_scale
        q = q.view(bsz, -1, self.action_num, self.num_atoms)
        q = q.squeeze(dim=-1)
        logits = q
        return logits, state
