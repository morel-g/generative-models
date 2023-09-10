import torch
from torch.optim.lr_scheduler import StepLR
from typing import Union, Tuple, Optional, Any, List, Dict, Type

from diffusers.optimization import get_cosine_schedule_with_warmup

from src.case import Case
from torch.nn import Module
from torch.optim import Optimizer


def create_optimizer(
    model: Module,
    training_params: Dict,
    optimizer_class: Type[Optimizer] = torch.optim.AdamW,
) -> torch.optim.Optimizer:
    """
    Create and return an optimizer.

    Parameters:
        model (Module): The PyTorch model for which the optimizer will be created.
        training_params (Dict[str, Union[float, int]]): Dictionary containing the training parameters.
            Must include keys 'lr' for learning rate and 'weight_decay' for weight decay.
        optimizer_class (Type[Optimizer], optional): Class of optimizer to create. Defaults to AdamW.

    Returns:
        Optimizer: The created optimizer.
    """
    if "lr" not in training_params or "weight_decay" not in training_params:
        raise ValueError(
            "Missing required keys in training_params: 'lr' and/or 'weight_decay'."
        )
    return optimizer_class(
        model.parameters(),
        lr=training_params["lr"],
        weight_decay=training_params["weight_decay"],
    )


def get_nb_steps_epoch(data_module, batch_size: int) -> int:
    """Calculate the number of steps in one epoch."""
    if data_module is not None:
        total_data = len(data_module.train_data)
        nb_steps_epoch = total_data // batch_size
        if total_data % batch_size != 0:
            nb_steps_epoch += 1
    else:
        nb_steps_epoch = 1
    return nb_steps_epoch


def create_scheduler(
    optimizer: Optimizer, training_params: Dict[str, Any], data_module=None
) -> Optional[Dict[str, Any]]:
    """
    Create and return a scheduler dictionary based on the training parameters.

    Parameters:
        optimizer: The optimizer for which to create a scheduler.
        training_params: Dictionary containing training parameters.
        data_module (optional): Data module containing training data.

    Returns:
        Dictionary containing scheduler information, or None if no scheduler type is provided.
    """
    scheduler_dict = training_params.get("scheduler_dict", {})
    scheduler_type = scheduler_dict.get("scheduler", None)

    if scheduler_type == Case.step_lr:
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_dict["every_n_epochs"],
            gamma=scheduler_dict["gamma"],
        )

    elif scheduler_type == Case.cosine_with_warmup:
        batch_size = training_params["batch_size"]
        nb_steps_epoch = get_nb_steps_epoch(data_module, batch_size)

        sch = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=scheduler_dict["num_warmup_epochs"]
            * nb_steps_epoch,
            num_training_steps=training_params["epochs"] * nb_steps_epoch,
        )
        scheduler = {
            "scheduler": sch,
            "interval": "step",
            "frequency": 1,
        }

    elif scheduler_type is None:
        scheduler = None

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler
