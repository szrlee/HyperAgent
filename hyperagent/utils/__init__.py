"""Utils package."""
import importlib

from hyperagent.utils.config import tqdm_config
from hyperagent.utils.logger.base import BaseLogger, LazyLogger
from hyperagent.utils.logger.tensorboard import BasicLogger, TensorboardLogger
from hyperagent.utils.logger.wandb import WandbLogger
from hyperagent.utils.statistics import MovAvg, RunningMeanStd

__all__ = [
    "MovAvg", "RunningMeanStd", "BaseLogger", "TensorboardLogger",
    "BasicLogger", "LazyLogger", "WandbLogger"
]


def import_module_or_data(import_path):
    try:
        maybe_module, maybe_data_name = import_path.rsplit(".", 1)
        print('trying from module {} import data {}'.format(maybe_module,
                                                            maybe_data_name))
        return getattr(importlib.import_module(maybe_module), maybe_data_name)
    except Exception as e:
        print('Cannot import data from the module path, error {}'.format(str(e)))

    try:
        print('trying to import module {}'.format(import_path))
        return importlib.import_module(import_path)
    except Exception as e:
        print('Cannot import module, error {}'.format(str(e)))

    raise ImportError('Cannot import module or data using {}'.format(import_path))


def read_config_dict(config_name):
    try:
        cfg = _read_config_dict_py_module(config_name)
        print('successfully _read_config_dict_py_module {}'.format(config_name),
                flush=True)
        return cfg
    except:
        pass

    try:
        cfg = _read_config_dict_py_expression(config_name)
        print('successfully _read_config_dict_py_expression {}'.format(config_name),
                flush=True)
        return cfg
    except:
        pass

    if config_name == "":
        print("Empty config string, returning empty dict", flush=True)
        return {}

    raise ValueError('Unknown cfg {}'.format(config_name))


def _read_config_dict_py_expression(config_name):
    return eval(config_name)


def _read_config_dict_py_module(config_name):
    config_name.lstrip("\"").rstrip("\"")
    config_module, config_name = config_name.rsplit(".", 1)
    config = getattr(importlib.import_module(config_module), config_name)
    return config
