from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.maze2d_umaze_v1
CONFIG["model_type"] = Case.score_model
CONFIG["scheme_params"].update(
    {
        "nb_time_steps_train": 1000,
        "nb_time_steps_eval": 1000,
        "conditioning_case": Case.conditioning_rl_first_last,
    }
)
CONFIG["training_params"].update(
    {
        "epochs": 2000,
        "check_val_every_n_epochs": 100,
        "batch_size": 64,
        "batch_size_eval": 64,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "ema_dict": {
            "use_ema": True,
            "decay": 0.995,
            "use_ema_warmup": True,
            "power": 2.0 / 3.0,
        },
        "gradient_clip_val": 1.0,
    }
)
CONFIG["model_params"] = {
    "transition_dim": 6,
    "horizon": 240,
    "dim": 32,
    "dim_mults": (1, 4, 8),
    "attention": False,
}
