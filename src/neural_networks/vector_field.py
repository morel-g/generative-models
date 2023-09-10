import torch
import torch.nn as nn

from src.utils import (
    relu_deriv,
    tanh_deriv,
    log_cosh,
    log_cosh_deriv,
)
from src.case import Case


class VectorField(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        nb_neurons=[],
        activation=Case.tanh,
        bias=True,
        gradient=False,
    ):
        """Vector field class.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            nb_neurons: A list with the number of neurons per layers. Defaults
            to [].
            activation: Non linear function apply to each layer choice between
            (Case.tanh, Case.relu, Case.log_cosh). Defaults to Case.tanh.
            bias: If bias are added to each layer. Defaults to True.
        """
        super(VectorField, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.gradient = gradient

        if gradient:
            if dim_out != dim_in:
                raise RuntimeError(
                    "The gradient computation is only implemented for"
                    + " vector fields with dim_in == dim_out"
                )
            self.dim_out = 1

        if self.activation == Case.log_cosh:
            self.deriv_func = log_cosh_deriv
            self.func = log_cosh
        elif self.activation == Case.relu:
            self.deriv_func = relu_deriv
            self.func = torch.relu
        elif self.activation == Case.tanh:
            self.deriv_func = tanh_deriv
            self.func = torch.tanh
        elif self.activation == Case.silu:
            self.deriv_func = None
            self.func = torch.nn.SiLU()
        else:
            raise RuntimeError(
                "Jacobian computation not implemented for activationation function"
                + self.func
            )

        self.neurons = nb_neurons.copy()
        self.neurons.insert(0, dim_in)
        self.neurons.append(dim_out)

        self.denses = nn.ModuleList(
            [
                nn.Linear(self.neurons[i], self.neurons[i + 1], bias=bias)
                for i in range(len(self.neurons) - 1)
            ]
        )

        [
            torch.nn.init.normal_(self.denses[i].weight, mean=0.0, std=1e-3)
            for i in range(len(self.denses))
        ]
        [
            torch.nn.init.zeros_(self.denses[i].bias) if bias else 0.0
            for i in range(len(self.denses))
        ]

    def __call__(self, x):
        if not self.gradient:
            return self.eval(x)
        else:
            return self.jacobian(x)

    def eval(self, x):
        """Evaluate the vecot field by iterating over the layers.

        Args:
            x: The positions.

        Returns:
            Vector field evaluated at x.
        """
        for i in range(len(self.denses)):
            d = self.denses[i]
            x = d(x)
            if i != len(self.denses) - 1:
                x = self.func(x)

        return x

    def jacobian(self, x):
        """Evaluate the jacobian of the vector field.

        Args:
            x: The positions.

        Returns:
            Jacobian evaluate at x.
        """
        deriv = self.deriv_func
        func = self.func

        batch = True if x.dim() != 1 else False
        for i in range(len(self.denses) - 1):
            d = self.denses[i]
            if i == 0:
                if not batch:
                    Jac = torch.einsum("j,jk->jk", deriv(d(x)), d.weight)
                else:
                    Jac = torch.einsum("ij,jk->ijk", deriv(d(x)), d.weight)

            else:
                if not batch:
                    Jac = torch.matmul(
                        torch.einsum("j,jk->jk", deriv(d(x)), d.weight), Jac
                    )
                else:
                    Jac = torch.matmul(
                        torch.einsum("ij,jk->ijk", deriv(d(x)), d.weight), Jac
                    )

            x = func(d(x))

        d = self.denses[-1]
        Jac = torch.matmul(d.weight, Jac)

        return Jac

    def gradient_eval(self, x):
        Jac = self.jacobian(x)
        return Jac.reshape(x.shape[0], x.shape[1])
