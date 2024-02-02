import torch
import torch.nn as nn

# from src.neural_networks.diffusion_model.u_net.old_u_net import UNet
from src.neural_networks.ncsnpp.ncsnpp import NCSNpp
from src.neural_networks.vector_field import VectorField
from src.neural_networks.u_net.u_net_2d_model import UNet2DModel
from src.neural_networks.u_net.u_net_1d_model import UNet1DModel
from src.neural_networks.u_net.u_net import UNet
from src.neural_networks.transformer_model import TransformerModel
from src.case import Case
from src.utils import time_dependent_var


class NeuralNetwork(torch.nn.Module):
    def __init__(self, model_type, config, stationary=False):
        super(NeuralNetwork, self).__init__()
        params_net = config.copy()
        self.stationary = stationary
        self.model_type = model_type
        if self.model_type == Case.ncsnpp:
            self.net = NCSNpp(**params_net)
        elif self.model_type == Case.u_net:
            self.net = UNet(**params_net)
        elif self.model_type == Case.u_net_1d:
            params_net.pop("horizon", None)
            self.net = UNet1DModel(**params_net)
        elif self.model_type == Case.transformer:
            params_net["batch_first"] = True
            self.net = TransformerModel(**params_net)
        elif self.model_type == Case.vector_field:
            if not self.stationary:
                params_net["dim_in"] += 1

            self.net = VectorField(**params_net)
        else:
            raise RuntimeError("Model type not implemented")

    def forward(self, x, t=None, t2=None):
        return self(x, t, t2)

    def __call__(self, x, t=None, t2=None):
        if self.model_type in (
            Case.u_net,
            Case.u_net_1d,
            Case.ncsnpp,
            Case.transformer,
        ):
            return self.net(x, t.view(-1)) if t is not None else self.net(x)
        else:
            if not self.stationary:
                x_t = time_dependent_var(x, t) if t is not None else x
                x_t = time_dependent_var(x_t, t2) if t2 is not None else x_t
            else:
                x_t = x
            return self.net(x_t)
