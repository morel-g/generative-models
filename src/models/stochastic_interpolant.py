import torch

from .model import Model
from ..case import Case
from .adapt_dt import adapt_dt_pdf


class StochasticInterpolant(Model):
    def __init__(
        self,
        data_type,
        model_params,
        nb_time_steps_eval,
        nb_time_steps_train=None,
        T_final=1.0,
        beta_case=Case.constant,
        adapt_dt=True,
        decay_case=Case.no_decay,
        interpolant=Case.linear,
        img_model_case=Case.u_net,
        noise_addition=None,
        exp_weight=1.0,
    ):
        self.beta_case = beta_case
        self.decay_case = decay_case
        self.interpolant = interpolant
        self.noise_addition = noise_addition

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

    def get_backward_schemes(self):
        return [Case.euler_explicit]

    def set_T_init(self, t_id=None, t_init=None, t_final=None):
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

    def normalize_time(self, t):
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

    def exponential_times_eval(self):
        return self.interpolant == Case.bgk

    def set_nb_time_steps(self, nb_time_steps, eval=False):
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
        pass

    def eval_nn(self, x, t):
        return self.neural_network(x, self.change_time_var(t))

    def velocity_eval(self, x, t):
        if self.decay_case == Case.no_decay:
            return self.eval_nn(x, t)
        elif self.decay_case == Case.exp:
            return self.eval_nn(x, t) * torch.exp(
                -self.exp_weight * self.change_time_var(t)
            )
        else:
            raise NotImplementedError(f"Unkwown decay case {self.decay_case}.")

    def conditional_forward_step(self, x, t1, t2, noise=None):
        # Not implemented yet
        return torch.zeros_like(x)

    def sample_time_uniform(self, t_shape, device):
        return torch.rand(t_shape, device=device) * self.T_final

    def sample_time_exp(self, t_shape, device):
        y = torch.rand(t_shape, device=device)
        C = 1 - self.exp_T
        return -torch.log(1 - C * y)

    def sample_time(self, x_shape, device):
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)
        if self.beta_case == Case.vanilla:
            return self.sample_time_uniform(t_shape, device) + self.T_init
        elif self.beta_case == Case.constant:
            return self.sample_time_exp(t_shape, device) + self.T_init
        else:
            raise NotImplementedError(f"Unkown beta {self.beta_case}")

    def eval_interpolant_path(self, x0, x1, t, interpolant):
        pi, Tf = torch.pi, self.T_final

        if interpolant == Case.bgk:
            t = t - self.T_init
            exp_t = torch.exp(-torch.tensor(t))
            It = exp_t * x0 + (1 - exp_t) * x1
            dt_It = -exp_t * x0 + exp_t * x1
        elif interpolant in (Case.linear, Case.poly):
            t = self.normalize_time(t)
            It = x0 * (1 - t) + t * x1
            dt_It = x1 - x0
            if self.add_noise:
                z = torch.randn_like(x0)
                time_noise = torch.sqrt(2 * t * (1 - t)) 
                It += time_noise * z
                dt_It += (2.-4.*t)*z/ (2.*time_noise)
                
        elif interpolant == Case.linear_scale:
            t = self.normalize_time(t)
            eps = 1e-5
            x0_coef, x1_coef = torch.sqrt(1 - t + eps), torch.sqrt(t + eps)
            It = x0 * x0_coef + x1_coef * x1
            dt_It = (x1 / x1_coef - x0 / x0_coef) / (2)  # * Tf)
        elif interpolant == Case.trigonometric:
            t = self.normalize_time(t)
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
            
        It_noise, dt_It_noise = self._compute_noise_addition(x0,t)
        It += It_noise
        dt_It += dt_It_noise
        
        return It, dt_It

    def _compute_noise_addition(self, x, t):
        if self.noise_addition is None:
            It = torch.zeros_like(x)
            dt_It = torch.zeros_like(x)
        elif self.noise_addition == Case.linear_noise:
            z = torch.randn_like(x)
            It = t*(1-t)*z
            dt_It = (1-2*t)*z
        elif self.noise_addition == Case.sqrt_noise:
            z = torch.randn_like(x)
            time_noise = torch.sqrt(2 * t * (1 - t))
            It = time_noise *z
            dt_It = (2.-4.*t)*z/ (2.*time_noise)
            
        return It, dt_It         

    def eval_path(self, x0, x1, t):
        return self.eval_interpolant_path(x0, x1, t, self.interpolant)

    def loss(self, x0, x1=None):
        t = self.sample_time(x0.shape, x0.device)
        if x1 is None:
            x1 = torch.randn_like(x0)
        It, dt_It = self.eval_path(x0, x1, t)
        v = self.velocity_eval(It, t)
        sum_dims = list(range(1, x0.dim()))
        return (-2.0 * dt_It * v + v**2).sum(sum_dims).mean()

    def integral_time_coef(self, t1, t2):
        """Integration in time of the coefficients $b$ from the Fokker Planck equation
        $$\partial_t p = b(t)\nabla \cdot (p * x + \nabla p) $$

        Args:
            t1: Initial time to integrate from.
            t2: Final time to integrate from.
        """

        dt = t2 - t1
        if self.beta_case == Case.constant:
            # Here g_1(t) = g_2(t) = gamma.
            return dt
        elif self.beta_case == Case.vanilla:
            # Here g_1(t) = g_2(t) = 0.5*(b_min + (1-b_min/T_final)*t).
            b_min, b_max = (
                0.1,
                20,
            )
            return b_min * dt / (2.0 * self.T_final) + 0.5 * (
                b_max - b_min
            ) * (t2**2 - t1**2) / (2.0 * self.T_final**2)
        else:
            raise RuntimeError("beta_case not implemented.")

    def velocity_step(self, x, t, dt, backward=False):
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
