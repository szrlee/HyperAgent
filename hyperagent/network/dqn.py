from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import BaseNet

ModuleType = Type[nn.Module]

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        prior_std: float = 0.0,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
    ):
        super().__init__()
        self.basedmodel = nn.Linear(in_features, out_features)
        if prior_std:
            self.priormodel = nn.Linear(in_features, out_features)
            for param in self.priormodel.parameters():
                param.requires_grad = False

        self.device = device
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, x: torch.Tensor, prior_x=None, ) -> Tuple[torch.Tensor, Any]:
        out = self.basedmodel(x)
        if prior_x is not None and self.prior_std > 0:
            prior_out = self.priormodel(prior_x)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out


class DQNNet(BaseNet):
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
        use_dueling: bool = False,
        base_prior: bool = True,
        model_type: str = 'mlp',
        normalize: str = '',
        embedding_dim: int = -1,
        last_layer_params: Dict[str, Any] = None
    ) -> None:
        last_layer = lambda x, y: Linear(x, y, **last_layer_params)

        super().__init__(
            state_shape, action_shape, hidden_sizes, norm_layer, activation, device,
            softmax, num_atoms, use_dueling, base_prior, model_type, normalize, embedding_dim,
            last_layer=last_layer,
        )

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        if self.model_type == 'mlp':
            s = s.reshape(-1, self.input_dim)
        hidden_logits, prior_hidden_logits = self.feature(s)
        bsz = hidden_logits.shape[0]
        q = self.Q(hidden_logits, prior_hidden_logits)
        q = q.view(bsz, self.action_num, self.num_atoms).squeeze(dim=-1)
        logits = q
        return logits, state

