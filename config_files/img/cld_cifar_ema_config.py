from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.cifar10
CONFIG["model_type"] = Case.score_model_critical_damped
CONFIG["scheme_params"]["decay_case"] = Case.vanilla_sigma
CONFIG["training_params"].update(
    {
        "epochs": 750,
        "batch_size": 128,
        "batch_size_eval": 256,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "ema_dict": {
            "use_ema": True,
            "decay": 0.9999,
            "use_ema_warmup": True,
            "power": 0.75,
        },
        "gradient_clip_val": 1.0,
        "check_val_every_n_epochs": 10,
    }
)
CONFIG["model_params"].update(
    {
        "embedding_type": Case.fourier,
        "n_channels": 128,
        "ch_mult": (1, 2, 2, 2),
        "n_resblocks": 4,
    }
)
