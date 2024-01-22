import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from datetime import datetime
from typing import Optional, Tuple

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
from src.params import Params


def load_params(params: Params) -> Params:
    """Loads the parameters from a checkpoint.

    Args:
        params (Params): Params object containing checkpoint information.

    Returns:
        Params: Loaded params with some attributes replaced by the input params's attributes.
    """
    ckpt_path = params.checkpoint_dict["training_ckpt_path"]
    load_path = ckpt_path[: ckpt_path.index("/Checkpoint")]
    data_load = load_obj(load_path + "/params.obj")

    data_load.checkpoint_dict = params.checkpoint_dict
    data_load.accelerator = params.accelerator
    data_load.device = params.device
    data_load.seed = params.seed
    return data_load


def setup_callbacks(params: Params, log_dir: str) -> list:
    """
    Set up the necessary callbacks for training.

    This function prepares the list of callbacks required for the training
    process. This includes setting up the checkpoint saving mechanism and
    optionally, the EMA (Exponential Moving Average) and learning rate monitor
    callbacks, based on the provided params settings.

    Args:
        params (Params): Params object which contains all necessary parameters
        including checkpoint settings, EMA settings,and others.
        logger_dir (str): The directory where the logs are saved.

    Returns:
        list: A list of initialized callbacks that will be used during training.

    Raises:
        RuntimeError: If 'ema' is set in the training parameters but 'ema_rate'
        is not specified.
    """

    val_loss_str = "val_loss"
    save_top_k = params.checkpoint_dict.get("save_top", 2)
    checkpoint_callback = ModelCheckpoint(
        monitor=val_loss_str,
        dirpath=log_dir,
        save_top_k=save_top_k,
        save_last=True,
        mode="min",
        filename="Checkpoint_{epoch}-{val_loss:.3f}",
        every_n_epochs=1,
    )

    callbacks = [checkpoint_callback]
    # if params.training_params.get("ema", False):
    #     if "ema_rate" not in pa   rams.training_params:
    #         raise RuntimeError("ema_rate not specified in training params.")
    #     ema = EMA(params.training_params["ema_rate"])
    #     callbacks.append(ema)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # if is_notebook():
    #     callbacks.append(TQDMNotebookProgressBar())

    return callbacks


def initialize_or_load_model(params: Params) -> DiffusionGenerator:
    """
    Initialize a new model or load an existing one from a checkpoint based on
    provided params settings.

    Args:
        params (Params): Params object which contains all necessary parameters
        including checkpoint settings and others.

    Returns:
        DiffusionGenerator: The network either freshly initialized or loaded
        from a checkpoint.
    """

    if not params.checkpoint_dict["restore_training"]:
        return DiffusionGenerator(params)

    params = (
        load_params(params)
        if params.checkpoint_dict.get("load_data", False)
        else params
    )
    return DiffusionGenerator.load_from_checkpoint(
        params.checkpoint_dict["training_ckpt_path"], params=params
    )


def train_model(
    params: Params,
    data_module: DataModule,
    logger: Optional[TensorBoardLogger] = None,
) -> DiffusionGenerator:
    """Main training function.

    Args:
        params (Params): Input params object.
        data_module (DataModule): Data module for loading training and validation datasets.
        logger (Optional[TensorBoardLogger], optional): Logger used to save the results. Defaults to None.

    Returns:
        DiffusionGenerator: The trained network.
    """
    if logger is None:
        logger = get_logger(params.logger_path)

    startTime = datetime.now()

    net = initialize_or_load_model(params)
    callbacks = setup_callbacks(params, logger.log_dir)

    # Set device for training
    device = params.device if params.accelerator == "gpu" else "auto"

    # Print and save params
    params.write(logger.log_dir + "/params.txt", should_print=True)

    # Initialize trainer and start training
    trainer = pl.Trainer(
        max_epochs=params.training_params["epochs"],
        accelerator=params.accelerator,
        devices=device,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=params.training_params["check_val_every_n_epochs"],
        enable_progress_bar=params.print_opt.get("enable_progress_bar", True),
        gradient_clip_val=params.training_params.get("gradient_clip_val", 0.0),
        reload_dataloaders_every_n_epochs=0,
        accumulate_grad_batches=params.training_params.get(
            "accumulate_grad_batches", 1
        ),
    )

    if not params.checkpoint_dict["restore_training"]:
        trainer.fit(net, datamodule=data_module)
    elif params.training_params["epochs"] > 0:
        trainer.fit(
            net,
            ckpt_path=params.checkpoint_dict["training_ckpt_path"],
            datamodule=data_module,
        )
    else:
        data_module.setup()

    # Load the best model checkpoint
    # if trainer.checkpoint_callback.best_model_path:
    #     print("Loading checkpoint " + trainer.checkpoint_callback.best_model_path)
    #     net = DiffusionGenerator.load_from_checkpoint(
    #         trainer.checkpoint_callback.best_model_path, params=params
    #     )

    print("Execution time =", datetime.now() - startTime)

    # Set the model device
    net_device = (
        torch.device("cuda") if params.accelerator == "gpu" else torch.device("cpu")
    )
    net.prepare_for_inference(data_module.train_dataloader(), net_device)

    return net


def run_sim(params: Params) -> Tuple[DiffusionGenerator, TensorBoardLogger]:
    """Runs the complete simulation including training and evaluation.

    Args:
        params (Params): Input params object.

    Returns:
        Tuple[DiffusionGenerator, TensorBoardLogger]: The trained network and logger used.
    """
    logger = get_logger(params.logger_path)
    save_obj(params, logger.log_dir + "/params.obj")

    data_module = DataModule(params, logger.log_dir)
    net = train_model(params, data_module, logger)

    # Evaluate the model
    net.eval()
    val_dataset = data_module.val_data
    with torch.no_grad():
        if params.data_type in toy_continuous_data_type:
            x_val = data_module.train_data.x
            compute_continuous_outputs_2d(net, x_val, logger.log_dir)
        elif params.data_type in toy_discrete_data_type:
            x_val = data_module.train_data.x
            compute_discrete_outputs_2d(net, x_val, logger.log_dir)
        elif params.data_type in img_data_type:
            compute_imgs_outputs(net, val_dataset, logger.log_dir, nb_rows=5, nb_cols=5)
        elif params.data_type in text_data_type:
            compute_text_outputs(net, val_dataset, logger.log_dir)
        elif params.data_type in rl_data_type:
            compute_rl_outputs(net, val_dataset, logger.log_dir)
        elif params.data_type in manifold_data_type:
            compute_manifold_outputs(net, val_dataset, logger.log_dir)
        else:
            raise RuntimeError(f"Uknown data_type {params.data_type}")

    return net, logger
