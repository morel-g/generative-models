from src.case import Case
from config_files.toy.default_toy_config import get_default_config

CONFIG = get_default_config()
CONFIG["model_type"] = Case.stochastic_interpolant
CONFIG["scheme_params"]["interpolant"] = Case.linear
CONFIG["scheme_params"]["beta_case"] = Case.constant
CONFIG["scheme_params"]["nb_time_steps_eval"] = 1
CONFIG["training_params"]["batch_size"] = 200
CONFIG["custom_data"] = {
    "use_custom_data": True,
    "data_dir": "/content/outputs/version_0/samples",
    "prefix_1": "samples_batch_",
    "prefix_2": "noises_batch_",
}
