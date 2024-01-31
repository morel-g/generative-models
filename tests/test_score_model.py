import sys

sys.path.append("..")
import unittest
import torch
import numpy as np
from scipy.integrate import quad
from src.models.helpers.adapt_dt import (
    exact_adapt_dt_constant_pde_coef,
    adapt_dt_pdf,
)
from optim.diffusion_generator import DiffusionGenerator
from src.data_manager.data_module import DataModule
from src.params import Params
from src.case import Case
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

DATA_DICT = {
    "n_samples": 400000,
    "data_type": Case.moons,  # Case.  #  #  Case.moons  #
    "model_params": {
        "dim": 2,
        "nb_neurons": [32] * 2,
        "activation": Case.tanh,
    },
    "scheme_params": {
        "T_final": 4.0,
        "nb_time_steps_train": 4 * 50,
        "nb_time_steps_eval": 50,
        "score_type": Case.singular_score,
        "adapt_dt": True,
        "gradient": not True,
        "pde_coefs": {
            "gamma": 1.0,
            "b_min": 0.1,
            "b_max": 20.0,
        },
        # Choice between constant and vanilla
        "beta_case": Case.constant,
        "decay_case": Case.exp,
        "interpolant": Case.poly,
        "loss_weight": Case.sigma,
        "loss_norm": Case.L2,
        "zeros_S0_vv": False,
        "sliced_score": not False,
    },
    "testing_params": {
        "sliced_score": True,
        "compute_full_interpolant": True,
        "discrete_interpolant": not True,
        "fixed_collisional_velocity": not True,
        "learn_omega_and_post_coll": not True,
        "generative_model": Case.score_model,
        "order_2": not True,
        "large_time_v": not True,
    },
    "custom_path_dict": {
        "use_custom_path": not True,
        "train_path": not False,
        "params": {
            "dim": 2,
            "nb_neurons": [32] * 2,
            "activation": Case.silu,
        },
        "nb_epochs_train_velocity": 5,
        "nb_epochs_train_path": 2,
        "train_1d_path": True,
    },
    "model_type": Case.score_model,  # _critical_damped_copy,
    "training_params": {
        "epochs": 20,
        "batch_size": 500,
        "batch_size_eval": 500,
        "lr": 5e-3,
        "weight_decay": 1e-3,
        "check_val_every_n_epochs": 20,
        "ema": not True,
        "ema_rate": 0.999,
        "gradient_clip_val": 0.0,
        "scheduler_dict": {
            "scheduler": Case.cosine_with_warmup,
            # LR scheduler params
            "gamma": 0.99,
            "every_n_epochs": 1,
            # Cosine with warmup params
            "num_warmup_epochs": 1,
        },
    },
    "checkpoint_dict": {
        "restore_training": False,
        "training_ckpt_path": "../outputs/",
        "load_data": False,
        "save_top": 5,
    },
    "logger_opt": {
        "logger_path": "../outputs/",
        "logger_case": Case.mlflow_logger,
        "kwargs": {"experiment_name": "ml_exp", "save_dir": "../mlruns"},
    },
    "accelerator": "cpu",
    "device": [0],
    # "precision": "32",
    "seed": torch.seed(),
}


# Define a simple dataset
class DiracDataset(Dataset):
    def __init__(self, x, num_samples=100):
        self.x = x
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x


class TestScoreModel(unittest.TestCase):
    def test_score(self):
        data = Params(**DATA_DICT)
        x_init = torch.randn(2)
        train_dataset = DiracDataset(x_init, num_samples=100)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        net = DiffusionGenerator(params)
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(net, train_dataloader)


if __name__ == "__main__":
    # unittest.main()
    params = Params(**DATA_DICT)
    x_init = torch.randn(2)
    train_dataset = DiracDataset(x_init, num_samples=100)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    net = DiffusionGenerator(params)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(net, train_dataloader)
