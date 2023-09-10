from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["model_type"] = Case.score_model_critical_damped
CONFIG["scheme_params"]["decay_case"] = Case.vanilla_sigma
