from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.cifar10
CONFIG["model_type"] = Case.score_model
CONFIG["scheme_params"]["beta_case"] = Case.vanilla
CONFIG["scheme_params"]["decay_case"] = Case.vanilla_sigma
CONFIG["training_params"].update(
    {
        "epochs": 800,
        "batch_size": 128,
        "batch_size_eval": 256,
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
