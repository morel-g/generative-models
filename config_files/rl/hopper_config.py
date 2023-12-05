from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.hopper_medium_v2
CONFIG["model_type"] = Case.score_model
CONFIG["scheme_params"].update(
    {
        "nb_time_steps_train": 1000,
        "nb_time_steps_eval": 1000,
    }
)
CONFIG["training_params"].update(
    {
        "epochs": 300,
        "check_val_every_n_epochs": 10,
        "batch_size": 256,
        "batch_size_eval": 256,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "ema_dict": {
            "use_ema": True,
            "decay": 0.999,
            "use_ema_warmup": True,
            "power": 2.0 / 3.0,
        },
        "gradient_clip_val": 1.0,
    }
)
CONFIG["model_params"] = {
    "transition_dim": 14,
    "horizon": 32,
    "dim": 64,
    "dim_mults": (1, 2, 2, 4),
    "attention": False,
}
