import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.logger import Logger
from datetime import datetime
from typing import Optional, Tuple
from omegaconf import OmegaConf

from src.utils import print_params
from src.data_manager.data_module import DataModule
from src.training.diffusion_generator import DiffusionGenerator
from src.utils import get_logger
from src.eval.plots_img import compute_imgs_outputs
from src.eval.plots_text import compute_text_outputs
from src.eval.plots_rl import compute_rl_outputs
from src.eval.plots_2d import (
    compute_continuous_outputs_2d,
    compute_discrete_outputs_2d,
)
from src.eval.plot_manifold import compute_manifold_outputs
from src.data_manager.data_type import (
    toy_continuous_data_type,
    toy_discrete_data_type,
    img_data_type,
    text_data_type,
    rl_data_type,
    manifold_data_type,
)
from src.save_load_obj import save_obj, load_obj
from omegaconf import DictConfig


def load_params(config: DictConfig) -> DictConfig:
    """Loads the parameters from a checkpoint.

    Args:
        config (DictConfig): DictConfig object containing checkpoint information.

    Returns:
        DictConfig: Loaded config with some attributes replaced by the input config's attributes.
    """
    ckpt_path = config.restore_ckpt_path
    load_path = ckpt_path[: ckpt_path.index("/Checkpoint")]
    data_load = load_obj(load_path + "/config.obj")

    data_load.checkpoint_dict = config.checkpoint_dict
    data_load.accelerator = config.accelerator
    data_load.device = config.device
    return data_load


def setup_callbacks(config: DictConfig, log_dir: str) -> list:
    """
    Set up the necessary callbacks for training.

    This function prepares the list of callbacks required for the training
    process. This includes setting up the checkpoint saving mechanism and
    optionally, the EMA (Exponential Moving Average) and learning rate monitor
    callbacks, based on the provided config settings.

    Args:
        config (DictConfig): DictConfig object which contains all necessary parameters
        including checkpoint settings, EMA settings,and others.
        logger_dir (str): The directory where the logs are saved.

    Returns:
        list: A list of initialized callbacks that will be used during training.

    Raises:
        RuntimeError: If 'ema' is set in the training parameters but 'ema_rate'
        is not specified.
    """

    val_loss_str = "val_loss"
    checkpoint_callback = ModelCheckpoint(
        monitor=val_loss_str,
        dirpath=log_dir,
        save_top_k=3,
        save_last=True,
        mode="min",
        filename="Checkpoint_{epoch}-{val_loss:.3f}",
        every_n_epochs=1,
    )

    callbacks = [checkpoint_callback]
    # if config.training_params.get("ema", False):
    #     if "ema_rate" not in pa   rams.training_params:
    #         raise RuntimeError("ema_rate not specified in training config.")
    #     ema = EMA(config.training_params["ema_rate"])
    #     callbacks.append(ema)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # if is_notebook():
    #     callbacks.append(TQDMNotebookProgressBar())

    return callbacks


def initialize_or_load_model(config: DictConfig) -> DiffusionGenerator:
    """
    Initialize a new model or load an existing one from a checkpoint based on
    provided config settings.

    Args:
        config (DictConfig): DictConfig object which contains all necessary parameters
        including checkpoint settings and others.

    Returns:
        DiffusionGenerator: The network either freshly initialized or loaded
        from a checkpoint.
    """

    if config.restore_ckpt_path is None:
        return DiffusionGenerator(config)

    return DiffusionGenerator.load_from_checkpoint(
        config.restore_ckpt_path, config=config
    )


def train_model(
    config: DictConfig,
    data_module: DataModule,
    logger: Optional[Logger] = None,
) -> DiffusionGenerator:
    """Main training function.

    Args:
        config (DictConfig): Input config object.
        data_module (DataModule): Data module for loading training and validation datasets.
        logger (Optional[Logger], optional): Logger used to save the results. Defaults to None.

    Returns:
        DiffusionGenerator: The trained network.
    """

    startTime = datetime.now()

    net = initialize_or_load_model(config)
    callbacks = setup_callbacks(config, logger.get_log_dir())

    # Set device for training
    device = config.device if config.accelerator == "gpu" else "auto"

    # Print and save configp
    print_params(config)
    OmegaConf.save(config=config, f=logger.get_log_dir() + "/config.yaml")

    # Initialize trainer and start training
    trainer = pl.Trainer(
        max_epochs=config.training_params["epochs"],
        accelerator=config.accelerator,
        devices=device,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=config.training_params["check_val_every_n_epochs"],
        enable_progress_bar=config.print_opt.get("enable_progress_bar", True),
        gradient_clip_val=config.training_params.get("gradient_clip_val", 0.0),
        reload_dataloaders_every_n_epochs=0,
        accumulate_grad_batches=config.training_params.get(
            "accumulate_grad_batches", 1
        ),
    )

    if config.restore_ckpt_path is None:
        trainer.fit(net, datamodule=data_module)
    elif config.training_params["epochs"] > 0:
        trainer.fit(
            net,
            ckpt_path=config.restore_ckpt_path,
            datamodule=data_module,
        )
    else:
        data_module.setup()

    # Load the best model checkpoint
    # if trainer.checkpoint_callback.best_model_path:
    #     print("Loading checkpoint " + trainer.checkpoint_callback.best_model_path)
    #     net = DiffusionGenerator.load_from_checkpoint(
    #         trainer.checkpoint_callback.best_model_path, config=config
    #     )

    print("Execution time =", datetime.now() - startTime)

    # Set the model device
    net_device = (
        torch.device("cuda") if config.accelerator == "gpu" else torch.device("cpu")
    )
    net.prepare_for_inference(data_module.train_dataloader(), net_device)

    return net


def run_sim(config: DictConfig) -> Tuple[DiffusionGenerator, Logger]:
    """Runs the complete simulation including training and evaluation.

    Args:
        config (DictConfig): Input config object.

    Returns:
        Tuple[DiffusionGenerator, Logger]: The trained network and logger used.
    """
    logger = get_logger(
        config.logger_opt["logger_path"],
        config.logger_opt["logger_case"],
        **config.logger_opt["kwargs"],
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    log_dir = logger.get_log_dir()
    save_obj(config, log_dir + "/config.obj")

    data_module = DataModule(config, log_dir)
    net = train_model(config, data_module, logger)

    # Evaluate the model
    net.eval()
    val_dataset = data_module.val_data
    with torch.no_grad():
        if config.data_type in toy_continuous_data_type:
            x_val = data_module.train_data.x
            compute_continuous_outputs_2d(net, x_val, log_dir)
        elif config.data_type in toy_discrete_data_type:
            x_val = data_module.train_data.x
            compute_discrete_outputs_2d(net, x_val, log_dir)
        elif config.data_type in img_data_type:
            compute_imgs_outputs(net, val_dataset, log_dir, nb_rows=5, nb_cols=5)
        elif config.data_type in text_data_type:
            compute_text_outputs(net, val_dataset, log_dir)
        elif config.data_type in rl_data_type:
            compute_rl_outputs(net, val_dataset, log_dir)
        elif config.data_type in manifold_data_type:
            compute_manifold_outputs(net, val_dataset, log_dir)
        else:
            raise RuntimeError(f"Uknown data_type {config.data_type}")

    return net, logger
