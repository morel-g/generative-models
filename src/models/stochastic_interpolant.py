import torch
from typing import Tuple, Optional

from src.models.model import Model
from src.case import Case
from src.models.adapt_dt import adapt_dt_pdf


class StochasticInterpolant(Model):
    """
    The stochastic interpolant class.
    """

    def __init__(
        self,
        data_type: str,
        model_params: dict,
        nb_time_steps_eval: int,
        nb_time_steps_train: Optional[int] = None,
        T_final: float = 1.0,
        beta_case: str = Case.constant,
        adapt_dt: bool = True,
        decay_case: str = Case.no_decay,
        interpolant: str = Case.linear,
        img_model_case: str = Case.u_net,
        noise_addition: Optional[str] = None,
        exp_weight: float = 1.0,
    ) -> None:
        """
        Initializes the StochasticInterpolant model.

        Parameters:
        - data_type (str): Type of the data.
        - model_params (dict): Dictionary of model parameters.
        - nb_time_steps_eval (int): Number of time steps for evaluation.
        - nb_time_steps_train (Optional[int]): Number of time steps for training.
        - T_final (float): Final time value.
        - beta_case (str): Choice for beta values. Default is Case.constant.
        - adapt_dt (bool): Whether to adapt delta time. Default is True.
        - decay_case (str): Choice for decay. Default is Case.no_decay.
        - interpolant (str): Type of interpolant. Default is Case.linear.
        - img_model_case (str): Image model case. Default is Case.u_net.
        - noise_addition (Optional[str]): Case for noise addition.
        - exp_weight (float): Exponential weight value. Default is 1.0.

        Returns:
        - None
        """

        self.beta_case = beta_case
        self.decay_case = decay_case
        self.interpolant = interpolant
        self.noise_addition = noise_addition
        T_final = self._get_final_time(T_final)

        super(StochasticInterpolant, self).__init__(
            data_type,
            model_params,
            nb_time_steps_eval,
            nb_time_steps_train,
            T_final=T_final,
            adapt_dt=adapt_dt,
            img_model_case=img_model_case,
        )
        self.backward_scheme = Case.euler_explicit
        self.exp_T = torch.exp(-torch.tensor(self.T_final))
        self.default_times_eval = self.times_eval.clone()
        self.T_init = 0.0
        self.exp_weight = exp_weight

    def _get_final_time(self, T_final: float) -> float:
        """
        Calculates the final time.

        Parameters:
        - T_final (float): Initial final time value.

        Returns:
        - float: Adjusted final time value.
        """

        if self.interpolant != Case.bgk:
            # Manually configure the final time to be 1.0
            return 1.0

        return T_final

    def set_T_init(
        self,
        t_id: Optional[float] = None,
        t_init: Optional[float] = None,
        t_final: Optional[float] = None,
    ) -> None:
        """
        Set the initial time (T_init).

        Parameters:
        - t_id (Optional[float]): Time id for calculation.
        - t_init (Optional[float]): Initial time.
        - t_final (Optional[float]): Final time.

        Returns:
        - None
        """

        if t_id is not None:
            T_init = self.T_final * t_id
            self.T_init = T_init
            self.times_eval = T_init + self.default_times_eval
        else:
            if t_init is None or t_final is None:
                raise RuntimeError("Wrong value for t_init or t_final")
            self.T_init = t_init
            if t_final is not None:
                self.T_final = t_final
            self.set_nb_time_steps(self.nb_time_steps_eval, eval=True)

    def normalize_time(self, t: float) -> float:
        """
        Normalizes the given time based on T_init and T_final.

        Parameters:
        - t (float): Time to be normalized.

        Returns:
        - float: Normalized time.
        """

        return (t - self.T_init) / self.T_final

    def change_time_var(self, t):
        """New time variable by making the change of variable t_new = C(t)
           where C is a primitive of the time dependent function in front
           of the pde.

        Args:
            t: The time.

        Returns:
            The new time variable.
        """
        if self.beta_case == Case.constant:
            t_new = t
        elif self.beta_case == Case.vanilla:
            b_min, b_max = (
                0.1,
                20,
            )
            t_new = b_min * t / (self.T_final * 2.0) + 0.5 * (
                b_max - b_min
            ) * (t**2) / (2.0 * self.T_final**2)
        else:
            raise RuntimeError("Unknown beta_case.")

        return t_new

    def exponential_times_eval(self) -> bool:
        """
        Check if times evaluation should be exponentially distributed.

        Parameters:
        - None

        Returns:
        - bool: True if the interpolant is Case.bgk, False otherwise.
        """
        return self.interpolant == Case.bgk

    def set_nb_time_steps(
        self, nb_time_steps: int, eval: bool = False
    ) -> None:
        """
        Set the number of time steps for evaluation or training.

        Parameters:
        - nb_time_steps (int): Number of time steps.
        - eval (bool): If True, sets for evaluation. Otherwise, for training.

        Returns:
        - None
        """
        Tf = self.T_final
        if eval:
            if self.exponential_times_eval():
                self.nb_time_steps_eval = nb_time_steps
                self.dt_eval, self.times_eval = adapt_dt_pdf(
                    lambda t: torch.exp(-self.change_time_var(t)),
                    self.nb_time_steps_eval,
                    self.T_init,
                    Tf,
                    start_left=False,
                    x0=None if self.beta_case != Case.vanilla else Tf / 2.0,
                )
            else:
                super(StochasticInterpolant, self).set_nb_time_steps(
                    nb_time_steps, eval=eval
                )
        else:
            self.nb_time_steps_train = nb_time_steps
            self.dt_train, self.times_train = self.compute_uniform_times(
                nb_time_steps,
                t0=self.T_init,
                t1=Tf,
            )

    def eval_nn(self, x, t: float) -> torch.Tensor:
        """
        Evaluate the neural network with changed time variable.

        Parameters:
        - x: Input data.
        - t (float): Time value.

        Returns:
        - torch.Tensor: Neural network output.
        """
        return self.neural_network(x, self.change_time_var(t))

    def velocity_eval(self, x, t: float) -> torch.Tensor:
        """
        Evaluate the velocity based on decay case.

        Parameters:
        - x: Input data.
        - t (float): Time value.

        Returns:
        - torch.Tensor: Velocity output.
        """
        if self.decay_case == Case.no_decay:
            return self.eval_nn(x, t)
        elif self.decay_case == Case.exp:
            return self.eval_nn(x, t) * torch.exp(
                -self.exp_weight * self.change_time_var(t)
            )
        else:
            raise NotImplementedError(f"Unkwown decay case {self.decay_case}.")

    def conditional_forward_step(
        self, x, t1: float, t2: float, noise=None
    ) -> torch.Tensor:
        """
        Compute a conditional forward step. (Not implemented yet)

        Parameters:
        - x: Input data.
        - t1 (float): Initial time.
        - t2 (float): Final time.
        - noise: Noise data (default is None).

        Returns:
        - torch.Tensor: Zero tensor of the same shape as x.
        """
        # Not implemented yet
        return torch.zeros_like(x)

    def sample_time_uniform(
        self, t_shape: Tuple[int], device: str
    ) -> torch.Tensor:
        """
        Sample time uniformly.

        Parameters:
        - t_shape (Tuple[int]): Shape for the sampled time.
        - device (str): Device type.

        Returns:
        - torch.Tensor: Sampled time tensor.
        """
        return torch.rand(t_shape, device=device) * self.T_final

    def sample_time_exp(
        self, t_shape: Tuple[int], device: str
    ) -> torch.Tensor:
        """
        Sample time exponentially.

        Parameters:
        - t_shape (Tuple[int]): Shape for the sampled time.
        - device (str): Device type.

        Returns:
        - torch.Tensor: Sampled time tensor.
        """
        y = torch.rand(t_shape, device=device)
        C = 1 - self.exp_T
        return -torch.log(1 - C * y)

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
        if self.beta_case == Case.vanilla:
            return self.sample_time_uniform(t_shape, device) + self.T_init
        elif self.beta_case == Case.constant:
            return self.sample_time_exp(t_shape, device) + self.T_init
        else:
            raise NotImplementedError(f"Unkown beta {self.beta_case}")

    def eval_interpolant_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        interpolant: str = Case.something,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Evaluate the interpolant path and its time derivative given two points, a time tensor, and an interpolant type.

        Parameters:
        - x0 (torch.Tensor): Initial point.
        - x1 (torch.Tensor): Final point.
        - t (torch.Tensor): Time tensor.
        - interpolant (str): Type of interpolation to use.

        Returns:
        - (torch.Tensor, torch.Tensor): Interpolant path and its time derivative.
        """
        pi = torch.pi

        if interpolant == Case.bgk:
            t = t - self.T_init
            exp_t = torch.exp(-torch.tensor(t))
            It = exp_t * x0 + (1 - exp_t) * x1
            dt_It = -exp_t * x0 + exp_t * x1
        elif interpolant in (Case.linear, Case.poly):
            It = x0 * (1 - t) + t * x1
            dt_It = x1 - x0

        elif interpolant == Case.linear_scale:
            eps = 1e-5
            x0_coef, x1_coef = torch.sqrt(1 - t + eps), torch.sqrt(t + eps)
            It = x0 * x0_coef + x1_coef * x1
            dt_It = (x1 / x1_coef - x0 / x0_coef) / (2)
        elif interpolant == Case.trigonometric:
            It = x0 * torch.cos(0.5 * pi * t) + torch.sin(0.5 * pi * t) * x1
            dt_It = (
                (-x0 * torch.sin(0.5 * pi * t) + torch.cos(0.5 * pi * t) * x1)
                * 0.5
                * pi
            )
        else:
            raise NotImplementedError(
                f"Unkown interpolant {self.interpolant}."
            )

        It_noise, dt_It_noise = self._compute_noise_addition(x0, t)
        It += It_noise
        dt_It += dt_It_noise

        return It, dt_It

    def _compute_noise_addition(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Compute the noise addition and its time derivative.

        Parameters:
        - x (torch.Tensor): Tensor to base the noise on.
        - t (torch.Tensor): Time tensor.

        Returns:
        - (torch.Tensor, torch.Tensor): Noise addition and its time derivative.
        """
        if self.noise_addition is None:
            It = torch.zeros_like(x)
            dt_It = torch.zeros_like(x)

        elif self.noise_addition == Case.linear_noise:
            z = self._get_noise_like(x)
            It = t * (1 - t) * z
            dt_It = (1 - 2 * t) * z

        elif self.noise_addition == Case.sqrt_noise:
            z = self._get_noise_like(x)
            time_noise = torch.sqrt(2 * t * (1 - t))
            It = time_noise * z
            dt_It = (2.0 - 4.0 * t) * z / (2.0 * time_noise)

        return It, dt_It

    def eval_path(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Evaluate the path using the current interpolant.

        Parameters:
        - x0 (torch.Tensor): Initial point.
        - x1 (torch.Tensor): Final point.
        - t (torch.Tensor): Time tensor.

        Returns:
        - (torch.Tensor, torch.Tensor): Interpolant path and its time derivative.
        """
        return self.eval_interpolant_path(x0, x1, t, self.interpolant)

    def loss(self, x0: torch.Tensor, x1: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the loss based on the interpolant path and velocity.

        Parameters:
        - x0 (torch.Tensor): Initial tensor.
        - x1 (torch.Tensor, optional): Final tensor. If not provided, noise is used.

        Returns:
        - torch.Tensor: Computed loss.
        """
        t = self.sample_time(x0.shape, x0.device)
        if x1 is None:
            x1 = self._get_noise_like(x0)
        It, dt_It = self.eval_path(x0, x1, t)
        v = self.velocity_eval(It, t)
        sum_dims = list(range(1, x0.dim()))

        return (-2.0 * dt_It * v + v**2).sum(sum_dims).mean()

    def integral_time_coef(self, t1: float, t2: float) -> float:
        """
        Integration in time of the coefficients.

        Parameters:
        - t1 (float): Initial time to integrate from.
        - t2 (float): Final time to integrate from.

        Returns:
        - float: Coefficient after integration.
        """
        dt = t2 - t1
        if self.beta_case == Case.constant:
            return dt
        elif self.beta_case == Case.vanilla:
            b_min, b_max = (
                0.1,
                20,
            )
            return b_min * dt / (2.0 * self.T_final) + 0.5 * (
                b_max - b_min
            ) * (t2**2 - t1**2) / (2.0 * self.T_final**2)
        else:
            raise RuntimeError("beta_case not implemented.")

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
        v = self.velocity_eval(x, t)

        if self.backward_scheme == Case.euler_explicit:
            int_coef = self.integral_time_coef(t, t + dt)
            int_coef = -int_coef if backward else int_coef

            x = x + int_coef * v
        else:
            raise RuntimeError(
                "Unknown backward scheme: ", self.backward_scheme
            )

        return x
