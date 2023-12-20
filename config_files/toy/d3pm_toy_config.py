from src.case import Case
from config_files.toy.default_toy_config import get_default_config

CONFIG = get_default_config()
CONFIG["model_type"] = Case.d3pm
CONFIG["data_type"] = Case.swissroll_discrete
CONFIG["model_params"] = {
    "nb_tokens": 50,
    "emb_dim": 32,
    "nb_heads": 2,
    "hidden_dim": 32,
    "nb_layers": 3,
    "dropout": 0.0,
}
CONFIG["scheme_params"].update(
    {
        "nb_time_steps_train": 400,
        "nb_time_steps_eval": 400,
        "transition_case": Case.uniform,
        "store_transition_matrices": False,
        "seq_length": 2,
    }
)
CONFIG["training_params"]["epochs"] = 200
CONFIG["training_params"]["check_val_every_n_epochs"] = 10
CONFIG["n_samples"] = 100000
