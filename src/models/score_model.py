import torch
from torch.nn import functional as F
from typing import Optional, Union, List

from src.models.model import Model
from src.case import Case
from src.models.helpers.adapt_dt import adapt_dt_pdf


class ScoreModel(Model):
    """
    A class representing the ScoreModel derived from the Model class.
    """

    def __init__(
        self,
        data_type: str,
        model_params: dict,
        nb_time_steps_eval: int,
        nb_time_steps_train: int = None,
        T_final: float = 1.0,
        T_init: float = 0.0,
        beta_case: str = Case.constant,
        adapt_dt: bool = True,
        pde_coefs: dict = {"gamma": 1.0},
        decay_case: str = Case.vanilla_sigma,
        img_model_case: str = Case.u_net,
        conditioning_case: Optional[str] = None,
    ) -> None:
        """
        Initializes an instance of the ScoreModel class.

        Parameters:
        - data_type (str): The type of data being used.
        - model_params (dict): Parameters for the model.
        - nb_time_steps_eval (int): Number of time steps for evaluation.
        - nb_time_steps_train (int): Number of time steps for training.
        - T_final (float): The final time value.
        - T_init (float): The initial time value.
        - beta_case (str): Beta case.
        - adapt_dt (bool): Whether to adapt dt or not.
        - pde_coefs (dict): Coefficients for the PDE.
        - decay_case (str): Decay case to be used in the velocity.
        - img_model_case (str): Model choice when dealing with images.

        Returns:
        - None
        """

        # Instance attributes
        self.beta_case = beta_case
        self.pde_coefs = pde_coefs
        self.decay_case = decay_case

        # Call the parent class's initializer
        super(ScoreModel, self).__init__(
            data_type,
            model_params,
            nb_time_steps_eval,
            nb_time_steps_train,
            T_final=T_final,
            T_init=T_init,
            adapt_dt=adapt_dt,
            img_model_case=img_model_case,
            conditioning_case=conditioning_case,
        )

        self.backward_scheme = Case.euler_explicit
        self.exp_T = torch.exp(-torch.tensor(self.T_final))
        self.exp_0 = torch.exp(-torch.tensor(self.T_init))

    def get_backward_schemes(self) -> List[str]:
        """
        Retrieve the available backward schemes.

        Returns:
        - List[str]: List of the supported backward schemes.
        """
        return [Case.euler_explicit, Case.anderson]

    def change_time_var(self, t: float) -> float:
        """
        Generate a new time variable by applying a change of variable transformation.

        Parameters:
        - t (float): Original time variable.

        Returns:
        - float: New time variable after the transformation.
        """
        if self.beta_case == Case.constant:
            gamma = self.pde_coefs["gamma"]
            t_new = gamma * t
        elif self.beta_case == Case.vanilla:
            b_min: float = 0.1
            b_max: float = 20.0
            t_new = b_min * t / (self.T_final * 2.0) + 0.5 * (b_max - b_min) * (
                t**2
            ) / (2.0 * self.T_final**2)
        else:
            raise RuntimeError("Unknown beta_case.")
        return t_new

    def integral_dt(self, t1: float, t2: float) -> float:
        """
        Integrate in time the coefficients 'b' from the Fokker Planck equation:
        ∂_t p = b(t)∇ · (p * x + ∇p).

        Parameters:
        - t1 (float): Initial time to start the integration.
        - t2 (float): Final time to end the integration.

        Returns:
        - float: Resultant integration of the coefficient over the time period.
        """
        return self.change_time_var(t2) - self.change_time_var(t2)

    def sigma_eval(
        self,
        t: torch.Tensor,
        t0: Optional[torch.Tensor] = None,
        double_precision: bool = False,
    ) -> torch.Tensor:
        """
        Evaluation of the standard deviation of the forward process.

        Parameters:
        - t (torch.Tensor): Time to evaluate the standard deviation.
        - t0 (Optional[torch.Tensor]): Initial time. If None, assume t0=0.
        Default is None.
        - double_precision (bool): If True, converts `t` to double precision.
        Default is False.

        Returns:
        - torch.Tensor: Evaluated standard deviation.
        """
        t = t.type(torch.float64) if double_precision else t
        t_new = self.change_time_var(t)
        if t0 is not None:
            t0_new = self.change_time_var(t0)
            t_new = t_new - t0_new

        return torch.sqrt(1 - torch.exp(-2 * t_new))

    def mean_eval(
        self, t: torch.Tensor, t0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluation of the mean of the forward process.

        Parameters:
        - t (torch.Tensor): Time to evaluate the mean.
        - t0 (Optional[torch.Tensor]): Initial time. If None,
        assume t0=0. Default is None.

        Returns:
        - torch.Tensor: Evaluated mean.
        """
        t_new = self.change_time_var(t)
        if t0 is not None:
            t0_new = self.change_time_var(t0)
            t_new = t_new - t0_new
        return torch.exp(-t_new)

    def set_nb_time_steps(self, nb_time_steps: int, eval: bool = False) -> None:
        """
        Set the number of time steps.

        Parameters:
        - nb_time_steps (int): Number of time steps.
        - eval (bool): If True, evaluate. Default is False.

        Returns:
        - None
        """
        Ti = self.T_init
        Tf = self.T_final
        if eval:
            if self.adapt_dt:
                self.nb_time_steps_eval = nb_time_steps
                self.dt_eval, self.times_eval = adapt_dt_pdf(
                    lambda t: torch.exp(-self.change_time_var(t)),
                    self.nb_time_steps_eval,
                    Ti,
                    Tf,
                    start_left=False,
                    x0=None if self.beta_case != Case.vanilla else Tf / 2.0,
                )
            else:
                super(ScoreModel, self).set_nb_time_steps(nb_time_steps, eval=eval)
        else:
            self.nb_time_steps_train = nb_time_steps
            self.dt_train, self.times_train = self.compute_uniform_times(
                nb_time_steps,
                t0=Ti,
                t1=Tf,
            )

    def eval_nn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the neural network.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - t (torch.Tensor): Time tensor.

        Returns:
        - torch.Tensor: Neural network output.
        """
        return self.neural_network(x, self.change_time_var(t))

    def velocity_eval(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the velocity.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - t (torch.Tensor): Time tensor.

        Returns:
        - torch.Tensor: Evaluated velocity.
        """
        if self.decay_case == Case.no_decay:
            return self.eval_nn(x, t)
        elif self.decay_case == Case.vanilla_sigma:
            sigma = self.sigma_eval(t)
            return self.eval_nn(x, t) / sigma
        elif self.decay_case == Case.exp:
            return self.eval_nn(x, t) * torch.exp(-self.change_time_var(t)) - x
        else:
            raise NotImplementedError(f"Unknown decay case {self.decay_case}.")

    def conditional_forward_step(
        self, x: torch.Tensor, t1: float, t2: float, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward step assuming the initial condition is a dirac located at x.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - t1 (float): Starting time.
        - t2 (float): Ending time.
        - noise (torch.Tensor, optional): Noise tensor. If not provided,
        random noise is generated.

        Returns:
        - torch.Tensor: Result of the forward step.
        """
        if noise is None:
            noise = torch.randn_like(x)
        mu = x * self.mean_eval(t2, t0=t1)
        sigma = self.sigma_eval(t2, t0=t1)
        return mu + sigma * noise

    def sample_from_array(
        self, times: torch.Tensor, t_shape: tuple, device: torch.device
    ) -> torch.Tensor:
        """
        Sample times from an array.

        Parameters:
        - times (torch.Tensor): Array of times.
        - t_shape (tuple): Desired shape of output tensor.
        - device (torch.device): Device to allocate tensor on.

        Returns:
        - torch.Tensor: Tensor containing sampled times.
        """
        times = times.to(device)
        t_id = torch.randint(0, self.nb_time_steps_train, t_shape, device=device).long()
        return times[t_id]

    def sample_uniform(
        self, t_shape: tuple, device: torch.device, apply_log: bool = False
    ) -> torch.Tensor:
        """
        Sample uniformly from a distribution with bound given by by the initial
        and final time of the model.

        Parameters:
        - t_shape (tuple): Desired shape of output tensor.
        - device (torch.device): Device to allocate tensor on.
        - apply_log (bool, optional): Flag to determine if log should be applied.
        Default is False.

        Returns:
        - torch.Tensor: Uniformly sampled tensor.
        """
        if not apply_log:
            return (
                torch.rand(t_shape, device=device) * (self.T_final - self.T_init)
                + self.T_init
            )
        else:
            exp_0, exp_T = self.exp_0.to(device), self.exp_T.to(device)
            uniform = torch.rand(t_shape, device=device)
            t = uniform * (exp_0 - exp_T) + exp_T
            return t

    def _sample_time(self, x_shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Sample time values based on the shape of x.

        Parameters:
        - x_shape (tuple): Shape of the input tensor x.
        - device (torch.device): Device to allocate tensor on.

        Returns:
        - torch.Tensor: Tensor with sampled time values.
        """
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)
        return self.sample_uniform(t_shape, device)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Computed loss value.
        """
        t = self._sample_time(x.shape, x.device)
        sigma = self.sigma_eval(t)
        noise = self._get_noise_like(x)
        x_n = self.mean_eval(t) * x + sigma * noise
        x_n = self._apply_conditioning(x_n, x, noise=noise)

        loss = F.mse_loss
        if self.decay_case == Case.vanilla_sigma:
            nn = self.eval_nn(x_n, t)
            return loss(-noise, nn)
        else:
            score = self.velocity_eval(x_n, t)
            return loss(-noise, sigma * score)

    def velocity_step(
        self, x: Union[float, int], t: float, dt: float, backward: bool = False
    ) -> Union[float, int]:
        """
        Wrapper function for classical_velocity_step to compute the velocity step.

        Parameters:
        - x (Union[float, int]): The input value.
        - t (float): Time parameter.
        - dt (float): Time increment.
        - backward (bool): Indicator for the backward calculation. Default is False.

        Returns:
        - Union[float, int]: The computed velocity step result.
        """
        return self.classical_velocity_step(x, t, dt, backward)

    def classical_velocity_step(
        self, x: Union[float, int], t: float, dt: float, backward: bool = False
    ) -> Union[float, int]:
        """
        Computes the classical velocity step based on the provided scheme.

        Parameters:
        - x (Union[float, int]): The input value.
        - t (float): Time parameter.
        - dt (float): Time increment.
        - backward (bool): Indicator for the backward calculation. Default is False.

        Returns:
        - Union[float, int]: The computed classical velocity step result.
        """
        score = self.velocity_eval(x, t)

        gamma = self.pde_coefs["gamma"]
        if self.backward_scheme == Case.euler_explicit:
            int_coef = self.integral_dt(t, t + dt)
            int_coef = -int_coef if backward else int_coef
            x = x - int_coef * (x + score)
        elif self.backward_scheme == Case.anderson:
            l = 1.0
            int_coef = self.integral_dt(t, t + dt)
            int_coef = -int_coef if backward else int_coef
            z = self._get_noise_like(x)
            if t[0] + 1e-7 > self.times_eval[2]:
                x = (
                    x
                    - int_coef * (x + (1 + l) * score)
                    + torch.sqrt(2 * l * abs(int_coef) * gamma) * z
                )
            else:
                x = x - int_coef * (x + score)
        else:
            raise RuntimeError("Unknown backward scheme: ", self.backward_scheme)

        return x
