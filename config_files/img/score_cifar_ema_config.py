from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.cifar10
CONFIG["model_type"] = Case.score_model
CONFIG["scheme_params"]["beta_case"] = Case.vanilla
CONFIG["scheme_params"]["decay_case"] = Case.vanilla_sigma
CONFIG["training_params"].update(
    {
        "epochs": 150,
        "batch_size": 128,
        "batch_size_eval": 256,
        "weight_decay": 0.0,
        "ema_dict": {"use_ema": True, "decay": 0.9999},
        "gradient_clip_val": 1.0,
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
CONFIG["checkpoint_dict"].update(
    {
        "restore_training": True,
        # The ema rate should be changed
        # 0.999 for paul and 0.9999 for pierre
        "training_ckpt_path": "../outputs/version_1/last.ckpt",
    }
)
