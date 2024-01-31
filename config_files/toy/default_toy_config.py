from src.case import Case
import torch

CONFIG = {
    "n_samples": 400000,
    "data_type": Case.swissroll,
    "model_params": {
        "dim": 2,
        "nb_neurons": [64] * 3,
        "activation": Case.silu,
    },
    "scheme_params": {
        "T_final": 3.0,
        "nb_time_steps_train": 1000,
        "nb_time_steps_eval": 1000,
        "adapt_dt": True,
        "decay_case": Case.no_decay,
    },
    "training_params": {
        "epochs": 200,
        "batch_size": 500,
        "batch_size_eval": 500,
        "lr": 5e-3,
        "weight_decay": 1e-3,
        "check_val_every_n_epochs": 20,
        "ema_dict": {"use_ema": False, "decay": 0.9999},
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
        "training_ckpt_path": "",
        "load_data": False,
        "save_top": 5,
    },
    "print_opt": {
        "enable_progress_bar": True,
    },
    "logger_opt": {
        "logger_path": "../outputs/",
        "logger_case": Case.mlflow_logger,
        "kwargs": {"experiment_name": "ml_exp", "save_dir": "../mlruns"},
    },
    "accelerator": "cpu",
    "device": [0],
}


def get_default_config():
    if torch.cuda.is_available():
        CONFIG["accelerator"] = "gpu"
        CONFIG["device"] = [0]
    else:
        CONFIG["accelerator"] = "cpu"

    return CONFIG
