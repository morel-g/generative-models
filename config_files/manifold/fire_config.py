from src.case import Case
from config_files.toy.default_toy_config import get_default_config

CONFIG = get_default_config()
CONFIG["model_type"] = Case.stochastic_interpolant
CONFIG["data_type"] = Case.fire
CONFIG["model_params"] = {
    "dim": 3,
    "nb_neurons": [128] * 5,
    "activation": Case.silu,
}
CONFIG["training_params"].update(
    {
        "epochs": 10000,
        "check_val_every_n_epochs": 1000,
        "batch_size": 128,
        "batch_size_eval": 128,
        "lr": 1e-3,
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
