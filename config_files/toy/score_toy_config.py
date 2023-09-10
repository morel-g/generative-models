from src.case import Case
from config_files.toy.default_toy_config import get_default_config

CONFIG = get_default_config()
CONFIG["model_type"] = Case.score_model
CONFIG["scheme_params"]["beta_case"] = Case.vanilla
CONFIG["scheme_params"]["decay_case"] = Case.vanilla_sigma
