from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import math
import numpy as np
import torch
from torch import nn

ModuleType = Type[nn.Module]

def dopamine_init(module):
    classname = module.__class__.__name__
    scale = 1.0 / np.sqrt(3.0)
    if hasattr(module, "weight"):
        if "Conv2d" in classname:
            fan_in = module.in_channels * np.prod(module.kernel_size)
        elif classname == "Linear":
            fan_in = module.in_features
        limit = np.sqrt((3 * scale) / fan_in)
        nn.init.uniform_(module.weight, a=-limit, b=limit)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def dqn_zoo_init(module):
    classname = module.__class__.__name__
    if "Conv2d" in classname:
        fan_in = module.in_channels * np.prod(module.kernel_size)
    elif classname == "Linear":
        fan_in = module.in_features
    else:
        return
    limit = np.sqrt(1 / fan_in)
    if hasattr(module, "weight"):
        nn.init.uniform_(module.weight, a=-limit, b=limit)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.uniform_(module.bias, a=-limit, b=limit)


def trunc_normal_init(module):
    classname = module.__class__.__name__
    if classname == "Linear":
        if hasattr(module, "weight"):
            bound = 1.0 / math.sqrt(module.in_features)
            nn.init.trunc_normal_(module.weight, std=bound, a=-2*bound, b=2*bound)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def xavier_uniform_init(module):
    classname = module.__class__.__name__
    if classname == "Linear":
        gain = 1.0
        if hasattr(module, "weight"):
            nn.init.xavier_uniform_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def xavier_normal_init(module):
    classname = module.__class__.__name__
    if classname == "Linear":
        gain = 1.0
        if hasattr(module, "weight"):
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


class MinMaxNorm(nn.Module):
    def __init__(self, first_dim=1):
        super().__init__()
        self.first_dim = first_dim

    def forward(self, x):
        flat_x = x.view(*x.shape[:self.first_dim], -1)
        max = torch.max(flat_x, self.first_dim, keepdim=True).values
        min = torch.min(flat_x, self.first_dim, keepdim=True).values
        flat_x = (flat_x - min)/(max - min + 1e-8)
        return flat_x.view(*x.shape)


class TFLinear(nn.Linear):
    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.trunc_normal_(self.weight, std=bound, a=-2*bound, b=2*bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(
            hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, activation_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(self, s: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            s = torch.as_tensor(
                s,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        return self.model(s.flatten(1))  # type: ignore


class Conv(nn.Module):
    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        hidden_size: int = 512,
        device: Optional[Union[str, int, torch.device]] = None,
    ):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            cnn_output_dim = int(np.prod(
                self.model(torch.zeros(1, c, h, w)).shape[1:]))
        self.model = nn.Sequential(
            self.model,
            nn.Linear(cnn_output_dim, hidden_size), nn.ReLU(inplace=True)
        )
        self.output_dim = hidden_size

    def forward(self, s: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            s = torch.as_tensor(
                s,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        return self.model(s)  # type: ignore


class Fc(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (),
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ):
        super().__init__()
        self.device = device
        model = [linear_layer(input_dim, hidden_sizes[0])]
        model += [nn.ReLU(inplace=True)]
        for i in range(1, len(hidden_sizes)):
            model += [linear_layer(hidden_sizes[i-1], hidden_sizes[i])]
            model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.output_dim = hidden_sizes[-1]

    def forward(self, s: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            s = torch.as_tensor(
                s,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        return self.model(s)  # type: ignore


class Net(nn.Module):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param bool softmax: whether to apply a softmax layer over the last layer's
        output.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param int num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param bool dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~hyperagent.network.common.MLP`. Default to None.

    .. seealso::

        Please refer to :class:`~hyperagent.network.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~hyperagent.network.continuous.Actor`,
        :class:`~hyperagent.network.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        logits = self.model(s)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class FeatureNet(nn.Module):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        linear_layer: Type[nn.Linear] = nn.Linear,
        model_type: str = 'mlp',
        prior_std: float = 0.,
        normalize: str ='',
        embedding_dim: int = -1,
    ):
        super().__init__()
        self.device = device
        self.prior_std = prior_std
        self.normalize = normalize
        if model_type == 'mlp':
            input_dim = int(np.prod(state_shape))
            if embedding_dim >= 0:
                embedding_dim = input_dim if embedding_dim == 0 else embedding_dim
                self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
                nn.init.orthogonal_(self.embedding.weight)
                self.embedding.weight.requires_grad = False
                input_dim = embedding_dim
            self.basedmodel = self._mlp(input_dim, hidden_sizes, linear_layer)
            if self.prior_std:
                self.priormodel = self._mlp(input_dim, hidden_sizes, linear_layer)
                for param in self.priormodel.parameters():
                    param.requires_grad = False
            self.output_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else input_dim
        elif model_type == 'conv':
            self.basedmodel = self._conv(*state_shape, hidden_sizes[0], linear_layer)
            if self.prior_std:
                self.priormodel = self._conv(*state_shape, hidden_sizes[0], linear_layer)
                for param in self.priormodel.parameters():
                    param.requires_grad = False
            self.output_dim = hidden_sizes[0]
        elif model_type == 'conv_v2':
            self.basedmodel = self._conv_v2(*state_shape, hidden_sizes[0], linear_layer)
            if self.prior_std:
                self.priormodel = self._conv_v2(*state_shape, hidden_sizes[0], linear_layer)
                for param in self.priormodel.parameters():
                    param.requires_grad = False
            self.output_dim = hidden_sizes[0]
        else:
            raise NotImplementedError(f'not model structure: {model_type}')

    def forward(self, s: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = s.astype(np.float32)
        elif isinstance(s, torch.Tensor):
            s = s.type(torch.float32)
        s = torch.as_tensor(s, device=self.device)
        if hasattr(self, "embedding"):
            s = self.embedding(s)
        logits = self.basedmodel(s)
        prior_logits = self.priormodel(s) if self.prior_std else None
        return logits, prior_logits

    def _mlp(self, input_dim, hidden_sizes, linear_layer):
        if len(hidden_sizes) == 0:
            return nn.Identity()
        model = [linear_layer(input_dim, hidden_sizes[0])]
        model += [nn.ReLU(inplace=True)]
        for i in range(1, len(hidden_sizes)):
            model += [linear_layer(hidden_sizes[i-1], hidden_sizes[i])]
            model += [nn.ReLU(inplace=True)]
        if self.normalize == 'Layer':
            model.append(nn.LayerNorm(hidden_sizes[-1]))
        elif self.normalize == 'MinMax':
            model.append(MinMaxNorm(1))
        model = nn.Sequential(*model)
        return model

    def _conv(self, c, h, w, hidden_size, linear_layer):
        model = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            cnn_output_dim = int(np.prod(
                model(torch.zeros(1, c, h, w)).shape[1:]))
        model_list = [model]
        if self.normalize == 'Layer':
            model_list.append(nn.LayerNorm(cnn_output_dim))
        elif self.normalize == 'MinMax':
            model_list.append(MinMaxNorm(1))
        model_list.extend([linear_layer(cnn_output_dim, hidden_size), nn.ReLU(inplace=True)])
        model = nn.Sequential(*model_list)
        return model

    def _conv_v2(self, c, h, w, hidden_size, linear_layer):
        model = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=5), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=5), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            cnn_output_dim = int(np.prod(
                model(torch.zeros(1, c, h, w)).shape[1:]))
        model_list = [model]
        if self.normalize == 'Layer':
            model_list.append(nn.LayerNorm(cnn_output_dim))
        elif self.normalize == 'MinMax':
            model_list.append(MinMaxNorm(1))
        model_list.extend([linear_layer(cnn_output_dim, hidden_size), nn.ReLU(inplace=True)])
        model = nn.Sequential(*model_list)
        return model


class BaseNet(nn.Module):
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
        normalize: str ='',
        embedding_dim: int = -1,
        last_layer: Any = None,
    ) -> None:
        super().__init__()
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.action_num = int(np.prod(action_shape))
        self.use_dueling = use_dueling
        self.model_type = model_type
        self.input_dim = int(np.prod(state_shape))
        self.feature = FeatureNet(
            state_shape, hidden_sizes, device, model_type=model_type, prior_std=base_prior,
            normalize=normalize, embedding_dim=embedding_dim,
        )
        self.q_input_dim = self.feature.output_dim
        self.q_output_dim = num_atoms * self.action_num
        self.Q = last_layer(self.q_input_dim, self.q_output_dim)
        if self.use_dueling:  # dueling
            self.v_input_dim = self.feature.output_dim
            self.v_output_dim = num_atoms
            self.V = last_layer(self.v_input_dim, self.v_output_dim)


class Recurrent(nn.Module):
    """Simple Recurrent network based on LSTM.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: s -> flatten -> logits.

        In the evaluation mode, s should be with shape ``[bsz, dim]``; in the
        training mode, s should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        s = self.fc1(s)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(
                s, (
                    state["h"].transpose(0, 1).contiguous(),
                    state["c"].transpose(0, 1).contiguous()
                )
            )
        s = self.fc2(s[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return s, {"h": h.transpose(0, 1).detach(), "c": c.transpose(0, 1).detach()}


class DataParallelNet(nn.Module):
    """DataParallel wrapper for training agent with multi-GPU.

    This class does only the conversion of input data type, from numpy array to torch's
    Tensor. If the input is a nested dictionary, the user should create a similar class
    to do the same thing.

    :param nn.Module net: the network to be distributed in different GPUs.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = nn.DataParallel(net)

    def forward(self, s: Union[np.ndarray, torch.Tensor], *args: Any,
                **kwargs: Any) -> Tuple[Any, Any]:
        if not isinstance(s, torch.Tensor):
            s = torch.as_tensor(s, dtype=torch.float32)
        return self.net(s=s.cuda(), *args, **kwargs)
