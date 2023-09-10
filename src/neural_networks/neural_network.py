import torch
import torch.nn as nn

# from src.neural_networks.diffusion_model.u_net.old_u_net import UNet
from src.neural_networks.ncsnpp.ncsnpp import NCSNpp
from .vector_field import VectorField
from src.neural_networks.u_net.u_net2 import UNet2DModel
from src.neural_networks.u_net.u_net import UNetFM
from src.case import Case
from src.utils import time_dependent_var


class NeuralNetwork(torch.nn.Module):
    def __init__(self, model_type, params, add_time=True):
        super(NeuralNetwork, self).__init__()
        params_net = params.copy()
        self.model_type = model_type
        if self.model_type == Case.u_net:
            self.net = UNet2DModel(**params_net)
        elif self.model_type == Case.ncsnpp:
            self.net = NCSNpp(**params_net)
        elif self.model_type == Case.u_net_fashion_mnist:
            self.net = UNetFM(**params_net)
        elif self.model_type == Case.vector_field:
            if add_time:
                params_net["dim_in"] += 1

            self.net = VectorField(**params_net)
        else:
            raise RuntimeError("Model type not implemented")

    def forward(self, x, t=None, t2=None):
        return self(x, t, t2)

    def __call__(self, x, t=None, t2=None):
        if self.model_type in (
            Case.u_net,
            Case.ncsnpp,
            Case.u_net_fashion_mnist,
        ):
            return self.net(x, t.view(-1)) if t is not None else self.net(x)
        else:
            x_t = time_dependent_var(x, t) if t is not None else x
            x_t = time_dependent_var(x_t, t2) if t2 is not None else x_t
            return self.net(x_t)


class MLP(nn.Module):
    def __init__(
        self, input_dim=2, index_dim=1, hidden_dim=128, augmented=False
    ):
        super().__init__()

        act = nn.SiLU()

        in_dim = (
            input_dim * 2 + index_dim if augmented else input_dim + index_dim
        )
        out_dim = input_dim

        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, u, t):
        h = torch.cat([u, t.reshape(-1, 1)], dim=1)
        output = self.main(h)

        return output


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim=2,
        index_dim=1,
        hidden_dim=64,
        n_hidden_layers=20,
        augmented=False,
    ):
        super().__init__()

        self.act = nn.SiLU()
        self.n_hidden_layers = n_hidden_layers
        in_dim = (
            input_dim * 2 + index_dim if augmented else input_dim + index_dim
        )
        out_dim = input_dim

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim + index_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim + index_dim, out_dim))

        self.layers = nn.ModuleList(layers)

    def _append_time(self, h, t):
        time_embedding = torch.log(t + 1e-5)
        return torch.cat([h, time_embedding.reshape(-1, 1)], dim=1)

    def forward(self, u, t):
        h0 = self.layers[0](self._append_time(u, t))
        h = self.act(h0)

        for i in range(self.n_hidden_layers):
            h_new = self.layers[i + 1](self._append_time(h, t))
            h = self.act(h + h_new)

        return self.layers[-1](self._append_time(h, t))
