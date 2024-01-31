from src.case import Case
import torch

CONFIG = {
    "data_type": Case.fashion_mnist,
    "model_params": {
        "ch_mult": (1, 2, 4),
        "n_channels": 16,
        "embedding_type": "no_embedding",
        "n_resblocks": 2,  #
        # "attn_resolutions": (14,),
    },
    "scheme_params": {
        "T_final": 3.0,
        "nb_time_steps_train": 1000,
        "nb_time_steps_eval": 1000,
        "adapt_dt": True,
        "beta_case": Case.constant,
        "decay_case": Case.no_decay,
        "img_model_case": Case.ncsnpp,
    },
    "model_type": Case.score_model,
    "training_params": {
        "epochs": 50,
        "batch_size": 64,
        "batch_size_eval": 64,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "check_val_every_n_epochs": 10,
        "ema_dict": {"use_ema": False},
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
    "accelerator": "gpu",
    "device": [0],
}


def get_default_config():
    if torch.cuda.is_available():
        CONFIG["accelerator"] = "gpu"
        CONFIG["device"] = [0]
    else:
        CONFIG["accelerator"] = "cpu"

    return CONFIG
