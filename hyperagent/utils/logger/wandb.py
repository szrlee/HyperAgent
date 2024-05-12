import argparse
import os
from typing import Callable, Optional, Tuple

from hyperagent.utils import BaseLogger
from hyperagent.utils.logger.base import LOG_DATA_TYPE

try:
    import wandb
except ImportError:
    pass


class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.

    This logger creates three panels with plots: train, test, and update.
    Make sure to select the correct access for each panel in weights and biases:

    - ``train/env_step`` for train plots
    - ``test/env_step`` for test plots
    - ``update/gradient_step`` for update plots

    Example of usage:
    ::

        with wandb.init(project="My Project"):
            logger = WandBLogger()
            result = onpolicy_trainer(policy, train_collector, test_collector,
                    logger=logger)

    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data().
        Default to 1000.
    :param str project: W&B project name. Default to "hyperagent".
    :param str name: W&B run name. Default to None. If None, random name is assigned.
    :param str entity: W&B team/organization name. Default to None.
    :param str run_id: run id of W&B run to be resumed. Default to None.
    :param argparse.Namespace config: experiment configurations. Default to None.
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1000,
        project: str = 'hyperagent',
        name: Optional[str] = None,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[argparse.Namespace] = None,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.last_save_step = -1
        self.save_interval = save_interval
        self.restored = False

        self.wandb_run = wandb.init(
            project=project,
            name=name,
            id=run_id,
            resume="allow",
            entity=entity,
            monitor_gym=True,
            config=config,  # type: ignore
        ) if not wandb.run else wandb.run
        self.wandb_run._label(repo="hyperagent")  # type: ignore

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        data[step_type] = step
        wandb.log(data)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)

            checkpoint_artifact = wandb.Artifact(
                'run_' + self.wandb_run.id + '_checkpoint',  # type: ignore
                type='model',
                metadata={
                    "save/epoch": epoch,
                    "save/env_step": env_step,
                    "save/gradient_step": gradient_step,
                    "checkpoint_path": str(checkpoint_path)
                }
            )
            checkpoint_artifact.add_file(str(checkpoint_path))
            self.wandb_run.log_artifact(checkpoint_artifact)  # type: ignore

    def restore_data(self) -> Tuple[int, int, int]:
        checkpoint_artifact = self.wandb_run.use_artifact(    # type: ignore
            'run_' + self.wandb_run.id + '_checkpoint:latest'  # type: ignore
        )
        assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist"

        checkpoint_artifact.download(
            os.path.dirname(checkpoint_artifact.metadata['checkpoint_path'])
        )

        try:  # epoch / gradient_step
            epoch = checkpoint_artifact.metadata["save/epoch"]
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = checkpoint_artifact.metadata["save/gradient_step"]
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = checkpoint_artifact.metadata["save/env_step"]
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0
        return epoch, env_step, gradient_step
