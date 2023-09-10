import torch
import numpy as np

from .model import Model
from ..case import Case
from .adapt_dt import adapt_dt_pdf, integrate
from torch.nn import functional as F
from scipy.integrate import quadrature

try:
    from torch.func import vjp
except ImportError:
    from functorch import vjp


class ScoreModel(Model):
    def __init__(
        self,
        data_type,
        model_params,
        nb_time_steps_eval,
        nb_time_steps_train=None,
        T_final=1.0,
        T_init=0.0,
        beta_case=Case.constant,
        adapt_dt=True,
        pde_coefs={"gamma": 1.0},
        decay_case=Case.vanilla_sigma,
        img_model_case=Case.u_net,
    ):
        self.beta_case = beta_case
        self.pde_coefs = pde_coefs
        self.decay_case = decay_case
        super(ScoreModel, self).__init__(
            data_type,
            model_params,
            nb_time_steps_eval,
            nb_time_steps_train,
            T_final=T_final,
            T_init=T_init,
            adapt_dt=adapt_dt,
            img_model_case=img_model_case,
        )
        self.backward_scheme = Case.euler_explicit
        self.exp_T = torch.exp(-torch.tensor(self.T_final))
        self.exp_0 = torch.exp(-torch.tensor(self.T_init))

    def get_backward_schemes(self):
        return [Case.euler_explicit, Case.anderson]

    def change_time_var(self, t):
        """New time variable by making the change of variable t_new = C(t)
        where C is a primitive of the time dependent function in front
        of the pde.
        """
        if self.beta_case == Case.constant:
            gamma = self.pde_coefs["gamma"]
            t_new = gamma * t
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

    def integral_time_coef(self, t1, t2):
        """Integration in time of the coefficients $b$ from the Fokker Planck equation
        $$\partial_t p = b(t)\nabla \cdot (p * x + \nabla p) $$

        Args:
            t1: Initial time to integrate from.
            t2: Final time to integrate from.
        """

        dt = t2 - t1
        if self.beta_case == Case.constant:
            # g_1(t) = g_2(t) = gamma.
            gamma = self.pde_coefs["gamma"]
            return dt * gamma
        elif self.beta_case == Case.vanilla:
            # g_1(t) = g_2(t) = 0.5*(b_min + (1-b_min/T_final)*t).
            b_min, b_max = (
                0.1,
                20,
            )
            return b_min * dt / (2.0 * self.T_final) + 0.5 * (
                b_max - b_min
            ) * (t2**2 - t1**2) / (2.0 * self.T_final**2)
        else:
            raise RuntimeError("beta_case not implemented.")

    def sigma_eval(self, t, t0=None, double_precision=False):
        """Evaluation of the std of the forward process.

        Args:
            t: time to evaluate the std.
            t0: Initial time if None assume t0=0. Default to None.
        """
        t = t.type(torch.float64) if double_precision else t
        t_new = self.change_time_var(t)
        if t0 is not None:
            t0_new = self.change_time_var(t0)
            t_new = t_new - t0_new

        return torch.sqrt(1 - torch.exp(-2 * t_new))

    def mean_eval(self, t, t0=None):
        """Evaluation of the mean of the forward process.

        Args:
            t: time to evaluate the mean.
            t0: Initial time if None assume t0=0. Default to None.
        """
        t_new = self.change_time_var(t)
        if t0 is not None:
            t0_new = self.change_time_var(t0)
            t_new = t_new - t0_new
        return torch.exp(-t_new)

    def set_nb_time_steps(self, nb_time_steps, eval=False):
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
                super(ScoreModel, self).set_nb_time_steps(
                    nb_time_steps, eval=eval
                )
        else:
            self.nb_time_steps_train = nb_time_steps
            self.dt_train, self.times_train = self.compute_uniform_times(
                nb_time_steps,
                t0=Ti,
                t1=Tf,
            )

    def eval_nn(self, x, t):
        return self.neural_network(x, self.change_time_var(t))

    def velocity_eval(self, x, t):
        if self.decay_case == Case.no_decay:
            return self.eval_nn(x, t)
        elif self.decay_case == Case.vanilla_sigma:
            sigma = self.sigma_eval(t)
            return self.eval_nn(x, t) / sigma
        elif self.decay_case == Case.exp:
            return self.eval_nn(x, t) * torch.exp(-self.change_time_var(t)) - x
        else:
            raise NotImplementedError(f"Unkwown decay case {self.decay_case}.")

    def conditional_forward_step(self, x, t1, t2, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        mu = x * self.mean_eval(t2, t0=t1)
        sigma = self.sigma_eval(t2, t0=t1)
        return mu + sigma * noise

    def exact_conditional_forward(self, x, t, noise):
        return self.conditional_forward_step(x, 0.0, t, noise)

    def sample_from_array(self, times, t_shape, device):
        times = times.to(device)
        t_id = torch.randint(
            0,
            self.nb_time_steps_train,
            t_shape,
            device=device,
        ).long()
        return times[t_id]

    def sample_uniform(self, t_shape, device, apply_log=False):
        if not apply_log:
            return (
                torch.rand(t_shape, device=device)
                * (self.T_final - self.T_init)
                + self.T_init
            )
        else:
            exp_0, exp_T = self.exp_0.to(device), self.exp_T.to(device)
            uniform = torch.rand(t_shape, device=device)
            t = uniform * (exp_0 - exp_T) + exp_T
            return t

    def sample_time(self, x_shape, device, L2_apply_log=True):
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)
        return self.sample_uniform(t_shape, device, False)

    def loss(self, x):
        t = self.sample_time(x.shape, x.device)

        noise = torch.randn_like(x)
        x_n = self.mean_eval(t) * x + self.sigma_eval(t) * noise
        nn = self.eval_nn(x_n, t)

        loss = F.mse_loss
        if self.decay_case == Case.vanilla_sigma:
            return loss(-noise, nn)
        else:
            score = self.velocity_eval(x_n, t)
            sigma = self.sigma_eval(t)
            return loss(
                -noise,
                sigma * score,
            )

    def velocity_step(self, x, t, dt, backward=False):
        return self.classical_velocity_step(x, t, dt, backward)

    def classical_velocity_step(self, x, t, dt, backward=False):
        score = self.velocity_eval(x, t)

        gamma = self.pde_coefs["gamma"]
        if self.backward_scheme == Case.euler_explicit:
            int_coef = self.integral_time_coef(t, t + dt)
            int_coef = -int_coef if backward else int_coef
            x = x - int_coef * (x + score)
        elif self.backward_scheme == Case.anderson:
            l = 1.0
            int_coef = self.integral_time_coef(t, t + dt)
            int_coef = -int_coef if backward else int_coef
            z = torch.randn_like(x)
            if t[0] + 1e-7 > self.times_eval[2]:
                x = (
                    x
                    - int_coef * (x + (1 + l) * score)
                    + torch.sqrt(2 * l * abs(int_coef) * gamma) * z
                )
            else:
                x = x - int_coef * (x + score)
        elif self.backward_scheme == Case.diffusion:
            one_over_mean = 1.0 / self.mean_eval(t)
            sigma = self.sigma_eval(t)

            cond_sigma_square = self.sigma_eval(t - dt, t) ** 2
            x = one_over_mean * (
                x + cond_sigma_square * score
            ) + sigma * torch.randn_like(x)
        else:
            raise RuntimeError(
                "Unknown backward scheme: ", self.backward_scheme
            )

        return x
