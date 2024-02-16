from typing import Optional, Dict, Union, Tuple

import torch

from src.models.model import Model
from src.case import Case
from src.models.helpers.adapt_dt import adapt_dt_pdf


class ScoreModelCriticalDamped(Model):
    """
    Score based critical damped Langevin model.
    """

    def __init__(
        self,
        data_type: str,
        model_params: dict,
        nb_time_steps_eval: int,
        nb_time_steps_train: int = None,
        T_final: float = 1.0,
        beta_case: str = Case.constant,
        adapt_dt: bool = True,
        decay_case: str = Case.vanilla_sigma,
        img_model_case: str = Case.u_net,
        init_var_v: float = 0.04,
        zeros_S0_vv: bool = False,
    ) -> None:
        """
        Initializes the ScoreModelCriticalDamped instance.

        Parameters:
        - data_type (str): The type of data for the model.
        - model_params (dict): The parameters for the model.
        - nb_time_steps_eval (int): The number of time steps for evaluation.
        - nb_time_steps_train (int, optional): The number of time steps for training.
        - T_final (float, optional): The final time value.
        - beta_case (str, optional): The beta case value.
        - adapt_dt (bool, optional): Whether to adapt dt.
        - decay_case (str, optional): The decay case value.
        - img_model_case (str, optional): The image model case value.
        - init_var_v (float, optional): The initial variance value.
        - zeros_S0_vv (bool, optional): Flag to determine if S0_vv=0.

        Returns:
        - None
        """
        self.beta_case = beta_case
        self.pde_coefs = {"gamma": 2.0}
        self.init_var_v = init_var_v
        self.zeros_S0_vv = zeros_S0_vv
        self.add_numerical_eps_to_sigma = not zeros_S0_vv
        self.numerical_eps = 1e-9
        self.decay_case = decay_case

        super(ScoreModelCriticalDamped, self).__init__(
            data_type,
            model_params,
            nb_time_steps_eval,
            nb_time_steps_train,
            T_final=T_final,
            adapt_dt=adapt_dt,
            img_model_case=img_model_case,
        )
        self.backward_scheme = Case.euler_explicit

    def get_initial_var(self) -> torch.Tensor:
        """
        Computes the initial variance.

        Parameters:
        - None

        Returns:
        - torch.Tensor: Initial variance value.
        """
        gamma = self.pde_coefs["gamma"]
        S0_vv = self.init_var_v * (gamma**2) / 4.0
        return torch.tensor(S0_vv)

    @staticmethod
    def _convert_tensor(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return t.type(dtype)

    # def cov_eval_vanilla(self, t, var0x=None, var0v=None):
    #     """
    #     Evaluating the variance of the conditional perturbation kernel.
    #     """
    #     t = t.type(torch.float64)
    #     beta = 1.0
    #     numerical_eps = 1e-9
    #     if var0x is None:
    #         var0x = torch.zeros_like(t, dtype=torch.float64)
    #         # var0x = add_dimensions(
    #         #     torch.zeros_like(t, dtype=torch.float64, device=t.device),
    #         #     False,
    #         # )
    #     gamma, m_inv = self.pde_coefs["gamma"], 4.0
    #     if var0v is None:
    #         var0v = self.get_initial_var() * torch.ones_like(
    #             t, dtype=torch.float64
    #         )
    #         # var0v = add_dimensions(var0v, False)
    #     g = 1.0 / torch.tensor(gamma, dtype=torch.float64)
    #     beta_int = beta * t  # self.beta_int_fn(t)
    #     f = torch.tensor(gamma, dtype=torch.float64)
    #     # beta_int = add_dimensions(self.beta_int_fn(t), False)
    #     multiplier = torch.exp(-4.0 * beta_int * g)
    #     var_xx = (
    #         var0x
    #         + (1.0 / multiplier)
    #         - 1.0
    #         + 4.0 * beta_int * g * (var0x - 1.0)
    #         + 4.0 * beta_int**2.0 * g**2.0 * (var0x - 2.0)
    #         + 16.0 * g**4.0 * beta_int**2.0 * var0v
    #     )
    #     var_xv = (
    #         -var0x * beta_int
    #         + 4.0 * g**2.0 * beta_int * var0v
    #         - 2.0 * g * beta_int**2.0 * (var0x - 2.0)
    #         - 8.0 * g**3.0 * beta_int**2.0 * var0v
    #     )
    #     var_vv = (
    #         f**2.0 * ((1.0 / multiplier) - 1.0) / 4.0
    #         + f * beta_int
    #         - 4.0 * g * beta_int * var0v
    #         + 4.0 * g**2.0 * beta_int**2.0 * var0v
    #         + var0v
    #         + beta_int**2.0 * (var0x - 2.0)
    #     )
    #     return [
    #         var_xx * multiplier + numerical_eps,
    #         var_xv * multiplier,
    #         var_vv * multiplier + numerical_eps,
    #     ]

    def cov_eval(
        self,
        t: torch.Tensor,
        t0: float = 0.0,
        return_ratio: bool = False,
        double_precision: bool = False,
        apply_log: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Evaluation of the standard deviation of the augmented forward process with S0_xx=0.

        Parameters:
        - t (torch.Tensor): Time to evaluate the std.
        - t0 (float): Initial time. Default is 0.0.
        - return_ratio (bool): If True, returns ratio in the results. Default is False.
        - double_precision (bool): If True, returns tensor with double precision. Default is False.
        - apply_log (bool): If True, applies log to the time variable. Default is False.

        Returns:
        - torch.Tensor or tuple of torch.Tensor: Standard deviations and optionally ratio.
        """

        t = t - t0
        return_type = torch.float64 if double_precision else torch.float32
        gamma = self.pde_coefs["gamma"]

        S0_vv = (
            torch.tensor(0.0, dtype=torch.float64)
            if self.zeros_S0_vv
            else self.get_initial_var() * torch.ones_like(t, dtype=torch.float64)
        )
        t = self._convert_tensor(t, torch.float64)
        exp_eval = torch.exp(-4 * t / gamma)

        if apply_log:
            t, exp_eval = -torch.log(t), t ** (4.0 / gamma)

        S_xx = (
            1
            + (
                -1
                - 4 * t / gamma
                - (8 * t**2) / (gamma**2)
                + (16 * t**2) * S0_vv / (gamma**4)
            )
            * exp_eval
        )
        S_xv = (
            4 * t * S0_vv / (gamma**2)
            + (4 * t**2) / gamma
            - 8 * (t**2) * S0_vv / (gamma**3)
        ) * exp_eval
        S_vv = (gamma**2) / 4.0 + (
            -(gamma**2) / 4.0
            + t * gamma
            + S0_vv * (1 + 4 * (t**2) / (gamma**2) - 4 * t / gamma)
            - 2 * (t**2)
        ) * exp_eval

        if self.add_numerical_eps_to_sigma:
            S_xx += self.numerical_eps
            S_vv += self.numerical_eps
            if return_ratio:
                ratio = S_xv**2 / S_xx

        elif return_ratio:
            eps_ratio = 1e-2
            ratio = S_xv**2 / S_xx
            if S0_vv > 0.0:
                ratio_limit = (
                    S0_vv
                    + ((gamma**2) * t**2) / (9 * S0_vv)
                    - (2 * (gamma**3) * t**3) / (27 * (S0_vv**2))
                    - (4 * S0_vv * t * (6 * (gamma**2) - 21 * gamma * t + 44 * (t**2)))
                    / (3 * (gamma**3))
                    + (2.0 / 15) * t * (10 * gamma - 55 * t + (136 * t**2) / gamma)
                )
            else:
                ratio_limit = 1.5 * gamma * t - 7.5 * t**2 + (18.3 * t**3) / gamma
            ratio = torch.where(t > eps_ratio, ratio, ratio_limit)

        S_xx, S_xv, S_vv = (
            self._convert_tensor(S_xx, return_type),
            self._convert_tensor(S_xv, return_type),
            self._convert_tensor(S_vv, return_type),
        )

        if return_ratio:
            ratio = self._convert_tensor(ratio, return_type)
            return S_xx, S_xv, S_vv, ratio

        return S_xx, S_xv, S_vv

    def sigma_eval(
        self,
        t: torch.Tensor,
        return_Svv: bool = False,
        double_precision: bool = False,
        apply_log: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate the standard deviation of the process.

        Parameters:
        - t (torch.Tensor): Time to evaluate the standard deviation.
        - return_Svv (bool): If True, returns S_vv along with the sigma. Default is False.
        - double_precision (bool): If True, returns tensor with double precision. Default is False.
        - apply_log (bool): If True, applies log to the time variable. Default is False.

        Returns:
        - torch.Tensor or tuple of torch.Tensor: Standard deviation sigma and optionally S_vv.
        """
        if self.decay_case == Case.vanilla_sigma:
            # if self.use_vanilla_cov:
            #     var0v = torch.tensor(0.0) if self.zeros_S0_vv else None
            #     S_xx, S_xv, S_vv = self.cov_eval_vanilla(
            #         t,
            #         var0v=var0v,
            #     )
            #     sigma = torch.sqrt(S_vv - S_xv**2 / S_xx)
            # else:
            S_xx, S_xv, S_vv = self.cov_eval(t, double_precision=True)
            sigma = torch.sqrt(S_vv - S_xv**2 / S_xx)
        else:
            S_xx, S_xv, S_vv, ratio = self.cov_eval(
                t,
                return_ratio=True,
                double_precision=True,
                apply_log=apply_log,
            )
            sigma = torch.sqrt(S_vv - ratio)

        if not double_precision:
            sigma = self._convert_tensor(sigma, torch.float32)
            S_vv = self._convert_tensor(S_vv, torch.float32)

        return sigma if not return_Svv else (sigma, S_vv)

    def mean_eval(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        v0: float = 0.0,
        t0: float = 0.0,
        apply_log: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluation of the mean of the forward process.

        Parameters:
        - t (torch.Tensor): Time to evaluate the mean.
        - x0 (torch.Tensor): Initial spatial point.
        - v0 (float): Initial velocity. Default is 0.0.
        - t0 (float): Initial time. Default is 0.0.
        - apply_log (bool): If True, applies log to the time variable. Default is False.

        Returns:
        - tuple of torch.Tensor: Mean values mu_1 and mu_2.
        """

        t = t - t0
        gamma = self.pde_coefs["gamma"]

        if not apply_log:
            exp_eval = torch.exp(-2 * t / gamma)
        else:
            t, exp_eval = -torch.log(t), t ** (2.0 / gamma)

        mu_1 = (x0 * (1 + 2 * t / gamma) + v0 * 4 * t / (gamma**2)) * exp_eval
        mu_2 = (-x0 * t + v0 * (-2 * t / gamma + 1)) * exp_eval

        return mu_1, mu_2

    def is_augmented(self) -> bool:
        """
        Determines if the current object is augmented.

        Returns:
        - bool: True if augmented, otherwise False.
        """
        return True

    def set_nb_time_steps(self, nb_time_steps: int, eval: bool = False) -> None:
        """
        Set the number of time steps.

        Parameters:
        - nb_time_steps (int): The number of time steps.
        - eval (bool): If True, the method will set evaluation-related attributes.
        Defaults to False.

        Returns:
        - None
        """
        if eval:
            self.nb_time_steps_eval = nb_time_steps
            gamma = self.pde_coefs["gamma"]

            self.dt_eval, self.times_eval = adapt_dt_pdf(
                lambda t: torch.exp(-(2.0 / gamma) * t),
                self.nb_time_steps_eval,
                0.0,
                self.T_final,
            )
        else:
            _, self.times_train = self.compute_uniform_times(
                nb_time_steps, 0.0, self.T_final
            )
            self.nb_time_steps_train = nb_time_steps

    def velocity_eval(self, x: torch.Tensor, v: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the velocity.

        Parameters:
        - x (torch.Tensor): The input tensor x.
        - v (torch.Tensor): The input tensor v.
        - t (float): The time.

        Returns:
        - torch.Tensor: The velocity evaluate at (x,v,t).
        """
        gamma = self.pde_coefs["gamma"]
        x_v = torch.cat((x, v), dim=1)
        sigma, S_vv = self.sigma_eval(t, return_Svv=True)
        inv_S_vv = 1.0 / S_vv

        if self.decay_case == Case.vanilla_sigma:
            if not self.zeros_S0_vv:
                return self.neural_network(x_v, t) / sigma - inv_S_vv * v
            else:
                return self.neural_network(x_v, t) / sigma - (4.0 / (gamma**2)) * v
        elif self.decay_case == Case.exp:
            return (
                self.neural_network(x_v, t) * torch.exp(-(2.0 / gamma) * t)
                - (4.0 / (gamma**2)) * v
            )
        elif self.decay_case == Case.vanilla:
            return self.neural_network(x_v, t)
        else:
            raise NotImplementedError(f"Unkwown decay case {self.decay_case}.")

    def conditional_forward_step(
        self,
        z: Tuple[torch.Tensor, torch.Tensor],
        t1: float,
        t2: float,
        noise: Optional[torch.Tensor] = None,
        apply_log: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a conditional forward step.

        Parameters:
        - z (Tuple[torch.Tensor, torch.Tensor]): A tuple containing tensors x and v.
        - t1 (float): Start time.
        - t2 (float): End time.
        - noise (Optional[torch.Tensor]): The noise tensor. Defaults to None.
        - apply_log (bool): If True, the logarithm will be applied. Defaults to False.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: The result of the forward step for x and v.
        """
        x, v = z
        noise_x = self._get_noise_like(x)
        noise_v = self._get_noise_like(x) if noise is None else noise
        mu_x, mu_v = self.mean_eval(t2, x0=x, v0=v, t0=t1, apply_log=apply_log)
        S_xx, S_xv, S_vv, ratio = self.cov_eval(
            t2,
            t0=t1,
            return_ratio=True,
            double_precision=True,
            apply_log=apply_log,
        )

        x = mu_x + torch.sqrt(S_xx) * noise_x
        v = mu_v + torch.sqrt(ratio) * noise_x + torch.sqrt(S_vv - ratio) * noise_v

        x = self._convert_tensor(x, torch.float32)
        v = self._convert_tensor(v, torch.float32)

        return (x, v)

    def exact_conditional_forward(
        self,
        x: torch.Tensor,
        t: float,
        noise: Optional[torch.Tensor] = None,
        apply_log: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the exact conditional forward step based on x.

        Parameters:
        - x (torch.Tensor): The input tensor x.
        - t (float): Time value.
        - noise (Optional[torch.Tensor]): The noise tensor. Defaults to None.
        - apply_log (bool): If True, the logarithm will be applied. Defaults to False.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: The result of the forward step for x and v.
        """
        noise_v = self._get_noise_like(x) if noise is None else noise
        v = torch.sqrt(self.get_initial_var()) * self._get_noise_like(x)
        z = x, v
        return self.conditional_forward_step(
            z,
            0.0,
            t,
            noise=noise_v,
            apply_log=apply_log,
        )

    def sample_time(self, x_shape: Tuple[int, ...], device: str) -> torch.Tensor:
        """
        Samples time tensor.

        Parameters:
        - x_shape (Tuple[int, ...]): Shape of the space x tensor.
        - device (str): Device to place the tensor on.

        Returns:
        - torch.Tensor: The sampled time tensor.
        """
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)
        return torch.rand(t_shape, device=device) * (self.T_final)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss.

        Parameters:
        - x (torch.Tensor): The input tensor x.

        Returns:
        - torch.Tensor: Computed loss.
        """
        t = self.sample_time(x.shape, x.device)
        gamma = self.pde_coefs["gamma"]
        noise = self._get_noise_like(x)
        x, v = self.exact_conditional_forward(x, t, noise)
        sigma, S_vv = self.sigma_eval(t, return_Svv=True)
        if self.decay_case == Case.vanilla_sigma and self.zeros_S0_vv:
            x_v = torch.cat((x, v), dim=1)
            loss = (
                self.neural_network(x_v, t) - sigma * (4.0 / (gamma**2)) * v + noise
            ) ** 2
        else:
            s = self.velocity_eval(x, v, t)
            loss = (s * sigma + noise) ** 2

        return torch.mean(torch.sum(loss.reshape(loss.shape[0], -1), dim=-1))

    def sample_prior_v(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples prior v.
        Samples from the prior distribution of the velocity variable, v.


        Parameters:
        - x (torch.Tensor): Input tensor, typically representing spatial points.

        Returns:
        - torch.Tensor: Sampled tensor.
        """
        gamma = self.pde_coefs["gamma"]
        v = (gamma / 2.0) * self._get_noise_like(x)
        return v

    def velocity_step(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        t: float,
        dt: float,
        backward: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the velocity step.

        Parameters:
        - x (Tuple[torch.Tensor, torch.Tensor]): Tuple containing tensors for x and v.
        - t (float): Time value.
        - dt (float): Delta time.
        - backward (bool): If True, the method performs backward steps. Defaults to False.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Resulting tensors after applying the velocity step.
        """
        gamma = self.pde_coefs["gamma"]

        x, v = x

        if backward:
            dt = -dt

        x = x - (dt / 2.0) * (-4.0 / gamma**2) * v
        s = self.velocity_eval(x, v, t + dt / 2.0)
        v = v - dt * (x + (4.0 / gamma) * v + gamma * s)
        x = x - (dt / 2.0) * (-4.0 / gamma**2) * v

        return (x, v)
