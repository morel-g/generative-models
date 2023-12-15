from src.case import Case
from config_files.toy.default_toy_config import get_default_config

CONFIG = get_default_config()
CONFIG["model_type"] = Case.stochastic_interpolant
CONFIG["scheme_params"]["interpolant"] = Case.linear
CONFIG["scheme_params"]["beta_case"] = Case.constant
CONFIG["custom_data"] = {
    "use_custom_data": True,
    "data_dir": "",
    "prefix_1": "samples_batch_",
    "prefix_2": "noises_batch_",
}
