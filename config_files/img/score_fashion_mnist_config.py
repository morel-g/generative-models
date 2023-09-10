from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["data_type"] = Case.fashion_mnist
CONFIG["model_type"] = Case.score_model
CONFIG["scheme_params"]["beta_case"] = Case.vanilla
CONFIG["scheme_params"]["decay_case"] = Case.vanilla_sigma
CONFIG["model_params"]["embedding_type"] = Case.fourier
