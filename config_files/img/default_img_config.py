from src.case import Case
import torch

CONFIG = {
    "data_type": Case.fashion_mnist,  # Case.multimodal_swissroll  #  #  Case.moons  #
    "model_params": {
        "ch_mult": (1, 2, 4),
        # "block_types": ("Block2D", "Block2D", "Block2D"),
        "n_channels": 16,
        "embedding_type": "no_embedding",  #  "linear",  #
        # "dropout": 0.0,
        "n_resblocks": 2,  #
        # "attn_resolutions": (14,),
    },
    "scheme_params": {
        "T_final": 3.0,
        "nb_time_steps_train": 200,
        "nb_time_steps_eval": 200,
        "adapt_dt": True,
        "beta_case": Case.constant,
        "decay_case": Case.no_decay,
        "img_model_case": Case.ncsnpp,  # n_net_fashion_mnist,  # ncsnpp,  #
    },
    "model_type": Case.score_model,  # _critical_damped,
    "training_params": {
        "epochs": 50,
        "batch_size": 32,
        "batch_size_eval": 32,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "check_val_every_n_epochs": 10,
        "ema": not True,
        "ema_rate": 0.99,
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
    "print_opt": {"ot_costs": False, "enable_progress_bar": True,},
    "logger_path": "../outputs/",
    "accelerator": "gpu",
    "device": [0],
    "seed": torch.seed(),
}


def get_default_config():
    if torch.cuda.is_available():
        CONFIG["accelerator"] = "gpu"
        CONFIG["device"] = [0]
    else:
        CONFIG["accelerator"] = "cpu"

    return CONFIG
