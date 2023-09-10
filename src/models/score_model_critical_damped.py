import torch
from .model import Model
from ..case import Case
from .adapt_dt import adapt_dt_pdf


class ScoreModelCriticalDamped(Model):
    def __init__(
        self,
        data_type,
        model_params,
        nb_time_steps_eval,
        nb_time_steps_train=None,
        T_final=1.0,
        beta_case=Case.constant,
        adapt_dt=True,
        decay_case=Case.vanilla_sigma,
        img_model_case=Case.u_net,
        init_var_v=0.04,
        zeros_S0_vv=False,
    ):
        if beta_case == Case.vanilla:
            T_final = 1.0
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

    def register_buffers(self):
        super(ScoreModelCriticalDamped, self).register_buffers()
        self.register_buffer("times_1", None)
        self.register_buffer("dt_1", None)
        self.register_buffer("times_2", None)
        self.register_buffer("dt_2", None)

    def get_initial_var(self):
        gamma = self.pde_coefs["gamma"]
        S0_vv = (
            self.init_var_v * (gamma**2) / 4.0
        )  # m*0.04 with m = gamma**2/4.
        return torch.tensor(S0_vv)

    def sample_init_v(self, shape, device):
        return torch.sqrt(self.get_initial_var()) * torch.randn(
            shape, device=device
        )

    def cov_eval_vanilla(self, t, var0x=None, var0v=None):
        """
        Evaluating the variance of the conditional perturbation kernel.
        """
        t = t.type(torch.float64)
        beta = 1.0
        numerical_eps = 1e-9
        if var0x is None:
            var0x = torch.zeros_like(t, dtype=torch.float64)
            # var0x = add_dimensions(
            #     torch.zeros_like(t, dtype=torch.float64, device=t.device),
            #     False,
            # )
        gamma, m_inv = self.pde_coefs["gamma"], 4.0
        if var0v is None:
            var0v = self.get_initial_var() * torch.ones_like(
                t, dtype=torch.float64
            )
            # var0v = add_dimensions(var0v, False)
        g = 1.0 / torch.tensor(gamma, dtype=torch.float64)
        beta_int = beta * t  # self.beta_int_fn(t)
        f = torch.tensor(gamma, dtype=torch.float64)
        # beta_int = add_dimensions(self.beta_int_fn(t), False)
        multiplier = torch.exp(-4.0 * beta_int * g)
        var_xx = (
            var0x
            + (1.0 / multiplier)
            - 1.0
            + 4.0 * beta_int * g * (var0x - 1.0)
            + 4.0 * beta_int**2.0 * g**2.0 * (var0x - 2.0)
            + 16.0 * g**4.0 * beta_int**2.0 * var0v
        )
        var_xv = (
            -var0x * beta_int
            + 4.0 * g**2.0 * beta_int * var0v
            - 2.0 * g * beta_int**2.0 * (var0x - 2.0)
            - 8.0 * g**3.0 * beta_int**2.0 * var0v
        )
        var_vv = (
            f**2.0 * ((1.0 / multiplier) - 1.0) / 4.0
            + f * beta_int
            - 4.0 * g * beta_int * var0v
            + 4.0 * g**2.0 * beta_int**2.0 * var0v
            + var0v
            + beta_int**2.0 * (var0x - 2.0)
        )
        return [
            var_xx * multiplier + numerical_eps,
            var_xv * multiplier,
            var_vv * multiplier + numerical_eps,
        ]

    def cov_eval(
        self,
        t,
        t0=0.0,
        S0_vv=torch.tensor(0.0),
        S0_xx=torch.tensor(0.0),
        return_ratio=False,
        double_precision=False,
        apply_log=False,
    ):
        """Evaluation of the std of the augmented forward process with S0_xx=0.

        Args:
            t: time to evaluate the std.
        """
        t = t - t0
        return_type = torch.float64 if double_precision else torch.float32
        gamma = self.pde_coefs["gamma"]
        # if S0_vv is None:
        #     S0_vv = self.get_initial_var()
        t, S0_vv = (
            t.type(torch.float64),
            S0_vv.type(torch.float64),
        )
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

        # eps = 1e-9
        # S_xx += self.numerical_eps
        # S_vv += self.numerical_eps
        if self.add_numerical_eps_to_sigma:
            S_xx += self.numerical_eps
            S_vv += self.numerical_eps
            if return_ratio:
                ratio = S_xv**2 / S_xx
                return (
                    S_xx.type(return_type),
                    S_xv.type(return_type),
                    S_vv.type(return_type),
                    ratio.type(return_type),
                )

        elif return_ratio:
            eps_ratio = 1e-2
            ratio = S_xv**2 / S_xx
            if S0_vv > 0.0:
                ratio_limit = (
                    S0_vv
                    + ((gamma**2) * t**2) / (9 * S0_vv)
                    - (2 * (gamma**3) * t**3) / (27 * (S0_vv**2))
                    - (
                        4
                        * S0_vv
                        * t
                        * (6 * (gamma**2) - 21 * gamma * t + 44 * (t**2))
                    )
                    / (3 * (gamma**3))
                    + (2.0 / 15)
                    * t
                    * (10 * gamma - 55 * t + (136 * t**2) / gamma)
                )
            else:
                ratio_limit = (
                    1.5 * gamma * t - 7.5 * t**2 + (18.3 * t**3) / gamma
                )
            ratio = torch.where(t > eps_ratio, ratio, ratio_limit)
            return (
                S_xx.type(return_type),
                S_xv.type(return_type),
                S_vv.type(return_type),
                ratio.type(return_type),
            )

        return (
            S_xx.type(return_type),
            S_xv.type(return_type),
            S_vv.type(return_type),
        )

    def sigma_eval(
        self,
        t,
        return_Svv=False,
        double_precision=False,
        # S0_vv=torch.tensor(0.0),
        apply_log=False,
    ):
        if self.decay_case == Case.vanilla_sigma:
            var0v = torch.tensor(0.0) if self.zeros_S0_vv else None
            S_xx, S_xv, S_vv = self.cov_eval_vanilla(
                t,
                var0v=var0v,
            )
            sigma = torch.sqrt(S_vv - S_xv**2 / S_xx)
        else:
            S0_vv = (
                torch.tensor(0.0)
                if self.zeros_S0_vv
                else self.get_initial_var()
                * torch.ones_like(t, dtype=torch.float64)
            )
            S_xx, S_xv, S_vv, ratio = self.cov_eval(
                t,
                S0_vv=S0_vv,
                return_ratio=True,
                double_precision=True,
                apply_log=apply_log,
            )
            sigma = torch.sqrt(S_vv - ratio)
        if not double_precision:
            sigma = sigma.type(torch.float32)
            S_vv = S_vv.type(torch.float32)
        return sigma if not return_Svv else (sigma, S_vv)

    def mean_eval(self, t, x0, v0=0.0, t0=0.0, apply_log=False):
        """Evaluation of the mean of the forward process.

        Args:
            t: time to evaluate the mean.
            x0: initial spatial point.
            v0: initial velocity. Default to 0.
            t0: initial time. Default to 0.
        """
        t = t - t0
        gamma = self.pde_coefs["gamma"]
        if not apply_log:
            exp_eval = torch.exp(-2 * t / gamma)
        else:
            t, exp_eval = -torch.log(t), t ** (2.0 / gamma)

        mu_1 = (
            x0 * (1 + 2 * t / gamma) + v0 * 4 * t / (gamma**2)
        ) * exp_eval
        mu_2 = (-x0 * t + v0 * (-2 * t / gamma + 1)) * exp_eval
        return mu_1, mu_2

    def is_augmented(self):
        return True

    def set_nb_time_steps(self, nb_time_steps, eval=False):
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

    def velocity_eval(self, x, v, t):
        gamma = self.pde_coefs["gamma"]
        x_v = torch.cat((x, v), dim=1)
        sigma, S_vv = self.sigma_eval(t, return_Svv=True)
        inv_S_vv = 1.0 / S_vv
        if self.decay_case == Case.vanilla_sigma:
            if not self.zeros_S0_vv:
                return self.neural_network(x_v, t) / sigma - inv_S_vv * v
            else:
                return (
                    self.neural_network(x_v, t) / sigma
                    - (4.0 / (gamma**2)) * v
                )
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
        self, z, t1, t2, noise=None, S0_vv=torch.tensor(0.0), apply_log=False
    ):
        x, v = z
        noise_x = torch.randn_like(x)
        noise_v = torch.randn_like(x) if noise is None else noise
        mu_x, mu_v = self.mean_eval(t2, x0=x, v0=v, t0=t1, apply_log=apply_log)
        S_xx, S_xv, S_vv, ratio = self.cov_eval(
            t2, t0=t1, S0_vv=S0_vv, return_ratio=True, apply_log=apply_log
        )

        x = mu_x + torch.sqrt(S_xx) * noise_x
        v = (
            mu_v
            + torch.sqrt(ratio) * noise_x
            + torch.sqrt(S_vv - ratio) * noise_v
        )

        return (x, v)

    def exact_conditional_forward(self, x, t, noise=None, apply_log=False):
        noise_v = torch.randn_like(x) if noise is None else noise
        v = torch.sqrt(self.get_initial_var()) * torch.randn_like(x)
        z = x, v
        S0_vv = (
            torch.tensor(0.0)
            if self.zeros_S0_vv
            else self.get_initial_var()
            * torch.ones_like(t, dtype=torch.float64)
        )
        return self.conditional_forward_step(
            z,
            0.0,
            t,
            noise=noise_v,
            S0_vv=S0_vv,
            apply_log=apply_log,
        )

    def sample_time(self, x_shape, device):
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)
        return torch.rand(t_shape, device=device) * (self.T_final)

    def loss(
        self,
        x,
    ):
        t = self.sample_time(x.shape, x.device)
        gamma = self.pde_coefs["gamma"]
        noise = torch.randn_like(x)
        x, v = self.exact_conditional_forward(x, t, noise)
        sigma, S_vv = self.sigma_eval(t, return_Svv=True)
        if self.decay_case == Case.vanilla_sigma and self.zeros_S0_vv:
            x_v = torch.cat((x, v), dim=1)
            loss = (
                self.neural_network(x_v, t)
                - sigma * (4.0 / (gamma**2)) * v
                + noise
            ) ** 2
        else:
            s = self.velocity_eval(x, v, t)
            loss = (s * sigma + noise) ** 2
        return torch.mean(torch.sum(loss.reshape(loss.shape[0], -1), dim=-1))

    def sample_prior_v(self, shape, device):
        gamma = self.pde_coefs["gamma"]
        v = (gamma / 2.0) * torch.randn(shape, device=device)
        return v

    def velocity_step(self, x, t, dt, backward=False):
        gamma = self.pde_coefs["gamma"]

        x, v = x

        if backward:
            dt = -dt

        x = x - (dt / 2.0) * (-4.0 / gamma**2) * v
        s = self.velocity_eval(x, v, t + dt / 2.0)
        v = v - dt * (x + (4.0 / gamma) * v + gamma * s)
        x = x - (dt / 2.0) * (-4.0 / gamma**2) * v

        return (x, v)
