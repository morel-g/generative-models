from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.lm1b_short
CONFIG["model_type"] = Case.d3pm
CONFIG["scheme_params"].update(
    {
        "nb_time_steps_train": 1000,
        "nb_time_steps_eval": 1000,
        "seq_length": 25,
        "tokenizer_name": Case.gpt2,
        "transition_case": Case.uniform,
    }
)
CONFIG["training_params"].update(
    {
        "epochs": 100,
        "check_val_every_n_epochs": 3,
        "batch_size": 64,
        "batch_size_eval": 64,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "ema_dict": {
            "use_ema": True,
            "decay": 0.9999,
            "use_ema_warmup": True,
            "power": 2.0 / 3.0,
        },
        "gradient_clip_val": 1.0,
    }
)
CONFIG["model_params"] = {
    "emb_dim": 512,
    "nb_heads": 8,
    "hidden_dim": 1024,
    "nb_layers": 4,
    "dropout": 0.1,
}
