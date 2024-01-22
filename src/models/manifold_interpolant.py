import torch
from typing import Tuple, Optional
from torch.autograd.functional import jvp
from geoopt.manifolds import Sphere

from src.models.model import Model
from src.case import Case
from src.models.helpers.adapt_dt import adapt_dt_pdf
from src.data_manager.data_type import manifold_data_type


class ManifoldInterpolant(Model):
    """
    The stochastic interpolant class.
    """

    def __init__(
        self,
        data_type: str,
        model_params: dict,
        nb_time_steps_eval: int,
        nb_time_steps_train: Optional[int] = None,
        noise_addition: Optional[str] = None,
    ) -> None:
        """
        Initializes the StochasticInterpolant model.

        Parameters:
        - data_type (str): Type of the data.
        - model_params (dict): Dictionary of model parameters.
        - nb_time_steps_eval (int): Number of time steps for evaluation.
        - nb_time_steps_train (Optional[int]): Number of time steps for training.
        - noise_addition (Optional[str]): Case for noise addition.

        Returns:
        - None
        """

        self.noise_addition = noise_addition

        super(ManifoldInterpolant, self).__init__(
            data_type,
            model_params,
            nb_time_steps_eval,
            nb_time_steps_train,
            T_final=1.0,
        )

        if self.data_type in [Case.earthquake, Case.fire, Case.flood]:
            self.manifold = Sphere()
        else:
            raise NotImplementedError(f"Manifold for data {data_type} not implemented.")

        self.backward_scheme = Case.euler_explicit

    def eval_nn(self, x, t: float) -> torch.Tensor:
        return self.neural_network(x, t)

    def velocity_eval(self, x, t: float) -> torch.Tensor:
        return self.eval_nn(x, t)

    def sample_prior_x(self, shape, device):
        return self.manifold.random_uniform(shape, dtype=torch.float, device=device)

    def conditional_forward_step(
        self, x, t1: float, t2: float, noise=None
    ) -> torch.Tensor:
        # Not implemented yet
        return torch.zeros_like(x)

    def sample_time_uniform(self, t_shape: Tuple[int], device: str) -> torch.Tensor:
        return torch.rand(t_shape, device=device) * self.T_final

    def sample_time(self, x_shape: Tuple[int], device: str) -> torch.Tensor:
        """
        Sample time based on the beta case.

        Parameters:
        - x_shape (Tuple[int]): Shape of the input data.
        - device (str): Device type.

        Returns:
        - torch.Tensor: Sampled time tensor.
        """
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)

        return self.sample_time_uniform(t_shape, device)

    def eval_path(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        tangent_vec = self.manifold.logmap(x0, x1)

        def expmap(t):
            return self.manifold.expmap(x0, t * tangent_vec)

        v = torch.ones_like(t)
        It, dt_It = jvp(expmap, (t,), (v,))
        return It, dt_It

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss based on the interpolant path and velocity.

        Parameters:
        - x (torch.Tensor or tuple of torch.Tensor): This parameter can be
        either a single torch.Tensor, representing the initial tensor, or a
        tuple of two torch.Tensor objects, representing the initial and final
        tensors respectively. If only the initial tensor is provided, noise is
        used for the final tensor.

        Returns:
        - torch.Tensor: Computed loss.
        """
        if isinstance(x, list):
            x0, x1 = x
        else:
            x0 = x
            x1 = self.sample_prior_x(x0.shape, x0.device)
        t = self.sample_time(x0.shape, x0.device)

        It, dt_It = self.eval_path(x0, x1, t)
        v = self.velocity_eval(It, t)
        sum_dims = list(range(1, x0.dim()))

        return (-2.0 * dt_It * v + v**2).sum(sum_dims).mean()

    def velocity_step(
        self, x: float, t: float, dt: float, backward: bool = False
    ) -> float:
        """
        Update the position based on the velocity over a time step.

        Parameters:
        - x (float): Current position.
        - t (float): Current time.
        - dt (float): Time step size.
        - backward (bool): If the step should be taken in the reverse direction.

        Returns:
        - float: Updated position after the step.
        """
        if not backward:
            # Not implemented
            return torch.zeros_like(x)

        v = self.velocity_eval(x, t)
        return self.manifold.expmap(x, -dt * v)
