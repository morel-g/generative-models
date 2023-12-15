from src.case import Case
from config_files.img.default_img_config import get_default_config

CONFIG = get_default_config()
CONFIG["model_type"] = Case.stochastic_interpolant
CONFIG["scheme_params"]["interpolant"] = Case.trigonometric
CONFIG["scheme_params"]["beta_case"] = Case.constant
CONFIG["model_params"]["embedding_type"] = Case.fourier
