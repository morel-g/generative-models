import torch
import numpy as np
import random

from src.models.model import Model
from src.case import Case
from src.models.adapt_dt import adapt_dt_pdf, integrate
from torch.nn import functional as F
from scipy.integrate import quadrature
from src.boltzman_sim import boltzman_step
from src.models.model import reduce_length_array


class EBMModel(Model):
    def __init__(
        self,
        data_type,
        model_params,
        nb_time_steps_eval,
        nb_time_steps_train,
        T_final=1.0,
        T_init=0.0,
        beta_case=Case.constant,
        adapt_dt=True,
        pde_coefs={"gamma": 1.0},
        decay_case=Case.exp,
        img_model_case=Case.u_net,
        batch_size=1,
        inter_it_dt=60,
    ):
        self.beta_case = beta_case
        self.pde_coefs = pde_coefs
        self.decay_case = decay_case
        super(EBMModel, self).__init__(
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
        self.inter_it_dt = inter_it_dt
        self.max_len = 8192
        self.examples = [
            [
                (torch.rand((1,) + self.particles_dim()) * 2 - 1)
                for _ in range(batch_size)
            ]
            for _ in range(nb_time_steps_train)
        ]

    def get_backward_schemes(self):
        return [Case.euler_explicit]

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
                super(EBMModel, self).set_nb_time_steps(
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

    def velocity_eval(self, x, t, stabilizing=True):
        model = self.neural_network
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        x.requires_grad = True
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        y = -self.eval_nn(x, t)
        y.sum().backward()
        # if stabilizing:
        #     x.grad.data.clamp_(
        #         -0.03, 0.03
        #     )  # For stabilizing and preventing too high gradients

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        return x.grad.data

    def sample_new_exmps(self, x_base, t_id, batch_size, device=None):
        """
        Function for getting a new batch of "fake" data.
        """

        if device is None:
            device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        t, dt = self.times_train[t_id], self.dt_train[t_id]
        t_id, dt = t_id[0], dt[0]
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(batch_size, 0.05)
        # rand_x = torch.randn((n_new,) + self.particles_dim()) * 2 - 1
        rand_indices = torch.randperm(x_base.shape[0])[:n_new]
        x_base_el = x_base[rand_indices]
        old_x = torch.cat(
            random.choices(self.examples[t_id], k=batch_size - n_new), dim=0
        )
        x = torch.cat([x_base_el, old_x], dim=0).detach().to(device)

        # Perform MCMC sampling
        x = self.classical_velocity_step(
            x, t, dt, backward=True, return_trajectories=False
        )

        # Add new images to the buffer and remove old ones if needed
        self.examples[t_id] = (
            list(x.to(torch.device("cpu")).chunk(batch_size, dim=0))
            + self.examples[t_id]
        )
        self.examples[t_id] = self.examples[t_id][: self.max_len]
        return x

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

    def sample_time(self, x_shape, device):
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)
        return self.sample_uniform(t_shape, device, False)

    def loss(self, x, x_base, t_id):
        # small_noise = torch.randn_like(x) * 0.005
        # x.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        t, dt = self.times_train[t_id], self.dt_train[t_id[0]]
        x_fake = self.sample_new_exmps(
            x_base, t_id, batch_size=x.shape[0], device=x.device
        )

        # Predict energy score for all points
        y = torch.cat([x, x_fake], dim=0)
        t = torch.cat([t, t], dim=0)
        y_real, y_fake = self.neural_network(y, t).chunk(2, dim=0)

        # Calculate losses
        reg_loss = (y_real**2 + y_fake**2).mean()
        cdiv_loss = y_fake.mean() - y_real.mean()
        return reg_loss, cdiv_loss, y_real, y_fake

    def loss_2(self, x_reals):
        # Switch the dimensions so that the time dependent dim is first
        x_reals.transpose_(0, 1)
        # Obtain samples
        t, dt = self.times_train[-1], self.dt_train[-1]
        x_fake = torch.randn_like(x_reals[-1])
        x_fakes = [x_fake]
        batch_size = x_fake.shape[0]
        times = [torch.full((batch_size,), t)]
        for t_id in reversed(range(self.nb_time_steps_train)):
            t_id = torch.full((batch_size,), t_id)
            x_fake = self.sample_new_exmps(
                x_fake, t_id, batch_size=x_fake.shape[0], device=x_fake.device
            )
            x_fakes.append(x_fake)
            times.append(self.times_train[t_id])
        x_fakes.reverse()
        times.reverse()
        x_fakes = torch.stack(x_fakes, dim=0)

        # Predict energy score for all points
        y = torch.cat(
            [
                x_reals.reshape(-1, *x_reals.shape[2:]),
                x_fakes.reshape(-1, *x_fakes.shape[2:]),
            ],
            dim=0,
        )
        times = 2 * times
        times = torch.cat(times, dim=0)
        y_real, y_fake = self.neural_network(y, times).chunk(2, dim=0)

        # Calculate losses
        reg_loss = (y_real**2 + y_fake**2).mean()
        cdiv_loss = y_fake.mean() - y_real.mean()
        return reg_loss, cdiv_loss, y_real, y_fake

    def conditional_forward_step(self, x, t1, t2):
        # return torch.zeros_like(x)
        return boltzman_step(x, t2 - t1)

    def velocity_step(
        self, x, t, dt, backward=False, return_trajectories=False
    ):
        return self.classical_velocity_step(
            x, t, dt, backward, return_trajectories
        )

    def classical_velocity_step(
        self, x, t, dt, backward=False, return_trajectories=True
    ):
        if not backward:
            # Not implemented
            return torch.zeros_like(x)

        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        steps = self.inter_it_dt
        Tf = 5.0
        step_size = Tf / steps
        dt_velocity = dt * step_size / Tf
        model = self.neural_network
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        x.requires_grad = True

        if return_trajectories:
            x_traj = [x.clone().detach().cpu().numpy()]

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Loop over K (steps)
        for k in range(steps):
            # Part 1: Add noise to the input.
            # noise.normal_(0, 0.005)
            # inp_imgs.data.add_(noise.data)
            # inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -self.eval_nn(x, t)  # + k * dt_velocity)
            out_imgs.sum().backward()
            # x.grad.data.clamp_(
            #     -0.03, 0.03
            # )  # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            x.data.add_(
                step_size * (-x.grad.data + np.sqrt(2.0) * torch.randn_like(x))
            )
            x.grad.detach_()
            x.grad.zero_()
            if return_trajectories:
                x_traj.append(x.clone().detach().cpu().numpy())
            # inp_imgs.data.clamp_(min=-1.0, max=1.0)

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        def trajectories_to_array(x_traj, is_augmented):
            if not is_augmented:
                return torch.tensor(np.array(x_traj))
            else:
                return torch.tensor(np.array(x_traj[0])), torch.tensor(
                    np.array(x_traj[1])
                )

        if return_trajectories:
            x_traj = trajectories_to_array(x_traj, False)
        return x_traj if return_trajectories else x

    def backward(
        self,
        x,
        return_trajectories=False,
        return_velocities=False,
        return_neural_net=False,
        save_score_pde_error=False,
    ):
        if return_velocities or return_neural_net:
            if return_velocities and return_neural_net:
                raise RuntimeError(
                    "return_velocities and return_neural_net cannot be True at the same time"
                )
            return np.array(
                self.compute_velocities(x, return_neural_net=return_neural_net)
            )
        elif self.backward_scheme == Case.RK45:
            return self.solve_score_RK45(
                x, backward=True, return_trajectories=return_trajectories
            )
        else:
            if save_score_pde_error:
                self.score_pde_error = []

            shape, dim = x.shape[0], x.dim()
            if self.is_augmented():
                v = self.sample_prior_v(x.shape, x.device)
                x = x, v
            else:
                x = x
            if return_trajectories:
                save_idx_times = self._get_traj_idx()
                x_traj = [] if not self.is_augmented() else ([], [])

                save_idx_times = save_idx_times[:-1]

            for i in reversed(range(self.nb_time_steps_eval)):
                dt = self.dt_eval[i]
                t = self.times_eval[i + 1].repeat((shape,) + (dim - 1) * (1,))
                if not return_trajectories:
                    x = self.velocity_step(
                        x.clone(),
                        t,
                        dt,
                        backward=True,
                        return_trajectories=return_trajectories,
                    )
                else:
                    x_traj.append(
                        self.velocity_step(
                            x.clone(),
                            t,
                            dt,
                            backward=True,
                            return_trajectories=return_trajectories,
                        )
                    )
                    x = x_traj[-1][-1]

            if return_trajectories:
                x_traj = reduce_length_array(torch.cat(x_traj, dim=0))
            return x_traj if return_trajectories else x
