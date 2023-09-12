from datetime import datetime
from typing import Optional, Tuple

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl
import torch

from src.data_manager.data_module import DataModule
from src.training.diffusion_generator import DiffusionGenerator
from src.utils import get_logger
from src.eval.plots import compute_imgs_outputs
from src.eval.plots_2d import compute_outputs_2d
from src.data_manager.data_type import toy_data_type
from src.save_load_obj import save_obj, load_obj
from src.training.ema import EMA
from src.data_manager.data import Data
from src.training.tqdm_notebook_progress_bar import TQDMNotebookProgressBar, is_notebook


def load_data(data: Data) -> Data:
    """Loads the data from a checkpoint.

    Args:
        data (Data): Data object containing checkpoint information.

    Returns:
        Data: Loaded data with some attributes replaced by the input data's attributes.
    """
    ckpt_path = data.checkpoint_dict["training_ckpt_path"]
    load_path = ckpt_path[: ckpt_path.index("/Checkpoint")]
    data_load = load_obj(load_path + "/data.obj")

    data_load.checkpoint_dict = data.checkpoint_dict
    data_load.accelerator = data.accelerator
    data_load.device = data.device
    data_load.seed = data.seed
    return data_load


def setup_callbacks(data: Data, log_dir: str) -> list:
    """
    Set up the necessary callbacks for training.

    This function prepares the list of callbacks required for the training
    process. This includes setting up the checkpoint saving mechanism and
    optionally, the EMA (Exponential Moving Average) and learning rate monitor
    callbacks, based on the provided data settings.

    Args:
        data (Data): Data object which contains all necessary parameters
        including checkpoint settings, EMA settings,and others.
        logger_dir (str): The directory where the logs are saved.

    Returns:
        list: A list of initialized callbacks that will be used during training.

    Raises:
        RuntimeError: If 'ema' is set in the training parameters but 'ema_rate'
        is not specified.
    """

    val_loss_str = "val_loss"
    save_top_k = data.checkpoint_dict.get("save_top", 2)
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
    if data.training_params.get("ema", False):
        if "ema_rate" not in data.training_params:
            raise RuntimeError("ema_rate not specified in training params.")
        ema = EMA(data.training_params["ema_rate"])
        callbacks.append(ema)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    if is_notebook():
        callbacks.append(TQDMNotebookProgressBar())

    return callbacks


def initialize_or_load_model(data: Data) -> DiffusionGenerator:
    """
    Initialize a new model or load an existing one from a checkpoint based on
    provided data settings.

    Args:
        data (Data): Data object which contains all necessary parameters
        including checkpoint settings and others.

    Returns:
        DiffusionGenerator: The network either freshly initialized or loaded
        from a checkpoint.
    """

    if not data.checkpoint_dict["restore_training"]:
        return DiffusionGenerator(data)

    data = (
        load_data(data)
        if data.checkpoint_dict.get("load_data", False)
        else data
    )
    return DiffusionGenerator.load_from_checkpoint(
        data.checkpoint_dict["training_ckpt_path"], data=data
    )


def train_model(
    data: Data,
    data_module: DataModule,
    logger: Optional[TensorBoardLogger] = None,
) -> DiffusionGenerator:
    """Main training function.

    Args:
        data (Data): Input data object.
        data_module (DataModule): Data module for loading training and validation datasets.
        logger (Optional[TensorBoardLogger], optional): Logger used to save the results. Defaults to None.

    Returns:
        DiffusionGenerator: The trained network.
    """
    if logger is None:
        logger = get_logger(data.logger_path)

    startTime = datetime.now()

    net = initialize_or_load_model(data)
    callbacks = setup_callbacks(data, logger.log_dir)

    # Set device for training
    device = data.device if data.accelerator == "gpu" else "auto"

    # Print and save data
    data.write(logger.log_dir + "/data.txt", print=True)

    # Initialize trainer and start training
    trainer = pl.Trainer(
        max_epochs=data.training_params["epochs"],
        accelerator=data.accelerator,
        devices=device,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=data.training_params[
            "check_val_every_n_epochs"
        ],
        enable_progress_bar=data.training_params["enable_progress_bar"],
        gradient_clip_val=data.training_params.get("gradient_clip_val", 0.0),
        reload_dataloaders_every_n_epochs=0,
    )

    if not data.checkpoint_dict["restore_training"]:
        trainer.fit(net, datamodule=data_module)
    elif data.training_params["epochs"] > 0:
        trainer.fit(
            net,
            ckpt_path=data.checkpoint_dict["training_ckpt_path"],
            datamodule=data_module,
        )
    else:
        data_module.setup()

    # Load the best model checkpoint
    if trainer.checkpoint_callback.best_model_path:
        print(
            "Loading checkpoint " + trainer.checkpoint_callback.best_model_path
        )
        net = DiffusionGenerator.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path, data=data
        )

    print("Execution time =", datetime.now() - startTime)

    # Set the model device
    net_device = (
        torch.device("cuda")
        if data.accelerator == "gpu"
        else torch.device("cpu")
    )
    net.to(net_device)

    return net


def run_sim(data: Data) -> Tuple[DiffusionGenerator, TensorBoardLogger]:
    """Runs the complete simulation including training and evaluation.

    Args:
        data (Data): Input data object.

    Returns:
        Tuple[DiffusionGenerator, TensorBoardLogger]: The trained network and logger used.
    """
    logger = get_logger(data.logger_path)
    save_obj(data, logger.log_dir + "/data.obj")

    data_module = DataModule(data, logger.log_dir)
    net = train_model(data, data_module, logger)

    # Evaluate the model
    net.eval()
    with torch.no_grad():
        if data.data_type in toy_data_type:
            x_val = data_module.train_data.x
            compute_outputs_2d(net, x_val, logger.log_dir)
        else:
            val_dataset = data_module.val_data
            compute_imgs_outputs(
                net, val_dataset, logger.log_dir, nb_rows=5, nb_cols=5
            )

    return net, logger
