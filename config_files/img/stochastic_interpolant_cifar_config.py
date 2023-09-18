from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.cifar10
CONFIG["model_type"] = Case.stochastic_interpolant
CONFIG["scheme_params"]["interpolant"] = Case.trigonometric
CONFIG["scheme_params"]["beta_case"] = Case.constant
CONFIG["training_params"].update(
    {
        "epochs": 150,
        "batch_size": 128,
        "batch_size_eval": 256,
        "ema": not True,
        "ema_rate": 0.99,
        "check_val_every_n_epochs": 10,
    }
)
CONFIG["model_params"].update(
    {
        "embedding_type": "positional",
        "n_channels": 128,
        "ch_mult": (1, 2, 2, 2),
        "n_resblocks": 4,
    }
)
