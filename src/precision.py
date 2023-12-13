import torch
import numpy as np


torch_float_precision = torch.float
if torch_float_precision == torch.double:
    eps_precision = np.finfo(np.float64).eps
else:
    eps_precision = np.finfo(np.float32).eps
