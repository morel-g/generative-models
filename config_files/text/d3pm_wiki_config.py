from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.wiki
CONFIG["model_type"] = Case.d3pm
CONFIG["scheme_params"].update(
    {
        "nb_time_steps_train": 1000,
        "nb_time_steps_eval": 1000,
        "seq_length": 25,
        "tokenizer_name": Case.gpt2,
    }
)
CONFIG["training_params"].update(
    {
        "epochs": 200,
        "check_val_every_n_epochs": 1,
        "batch_size": 64,
        "batch_size_eval": 64,
    }
)
CONFIG["model_params"] = {
    "emb_dim": 16,
    "nb_heads": 8,
    "hidden_dim": 16,
    "nb_layers": 4,
    "dropout": 0.1,
}
CONFIG["checkpoint_dict"].update(
    {
        "restore_training": False,
        "training_ckpt_path": "../outputs/version_24/last.ckpt",
    }
)
