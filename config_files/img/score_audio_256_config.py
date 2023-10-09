from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.audio_diffusion_256
CONFIG["model_type"] = Case.score_model
CONFIG["scheme_params"].update(
    {
        "beta_case": Case.vanilla,
        "decay_case": Case.vanilla_sigma,
        "nb_time_steps_eval": 200,
    }
)
CONFIG["training_params"].update(
    {
        "epochs": 100,
        "batch_size": 4,
        "batch_size_eval": 8,
        "check_val_every_n_epochs": 2,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "accumulate_grad_batches": 4,
        "ema": True,
        "ema_rate": 0.9999,
        "gradient_clip_val": 1.0,
    }
)
CONFIG["model_params"].update(
    {
        "embedding_type": Case.fourier,
        "n_channels": 128,
        "ch_mult": (1, 1, 2, 2, 4, 4),
        "n_resblocks": 2,
    }
)
