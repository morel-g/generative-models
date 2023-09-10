# Denoising model from https://huggingface.co/blog/annotated-diffusion
import torch
from torch.nn import functional as F
import numpy as np

from .time_schedule import linear_beta_schedule
from .denoise_helpers import q_sample, p_sample
from ...case import Case
from ...data_manager.data_type import toy_data_type
from ..model import Model
from ...utils import t_from_id


class DenoiseModel(Model):
    def __init__(
        self,
        data_type,
        model_params,
        nb_time_steps_eval,
        nb_time_steps_train,
        T_final=200,
        img_model_case=Case.u_net,
    ):
        super(DenoiseModel, self).__init__(
            data_type,
            model_params,
            nb_time_steps_eval,
            nb_time_steps_train,
            T_final=T_final,
            img_model_case=img_model_case,
        )
        self.set_diffusion_arrays()
        self.set_diffusion_arrays(eval=True)

    def set_diffusion_arrays(self, eval=False):
        nb_time_steps = (
            self.nb_time_steps_train if not eval else self.nb_time_steps_eval
        )
        betas = linear_beta_schedule(timesteps=nb_time_steps)

        # define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        # Put a 1. in the first position and move everything else one position to the right.
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        if not eval:
            self.betas = betas
            self.alphas = alphas
            self.alphas_cumprod = alphas_cumprod
            self.alphas_cumprod_prev = alphas_cumprod_prev
            self.sqrt_recip_alphas = sqrt_recip_alphas
            self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
            self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
            self.posterior_variance = posterior_variance
        else:
            self.betas_eval = betas
            self.alphas_eval = alphas
            self.alphas_cumprod_eval = alphas_cumprod
            self.alphas_cumprod_prev_eval = alphas_cumprod_prev
            self.sqrt_recip_alphas_eval = sqrt_recip_alphas
            self.sqrt_alphas_cumprod_eval = sqrt_alphas_cumprod
            self.sqrt_one_minus_alphas_cumprod_eval = (
                sqrt_one_minus_alphas_cumprod
            )
            self.posterior_variance_eval = posterior_variance

    def register_buffers(self):
        super(DenoiseModel, self).register_buffers()
        for eval_str in ["", "_eval"]:
            self.register_buffer("betas" + eval_str, None)
            self.register_buffer("alphas" + eval_str, None)
            self.register_buffer("alphas_cumprod" + eval_str, None)
            self.register_buffer("alphas_cumprod_prev" + eval_str, None)
            self.register_buffer("sqrt_recip_alphas" + eval_str, None)
            self.register_buffer("sqrt_alphas_cumprod" + eval_str, None)
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod" + eval_str,
                None,
            )
            self.register_buffer("posterior_variance" + eval_str, None)

    def set_nb_time_steps(self, nb_time_steps, eval=False):
        super(DenoiseModel, self).set_nb_time_steps(nb_time_steps, eval=eval)
        if not eval:
            self.set_diffusion_arrays(eval=False)
        else:
            self.set_diffusion_arrays(eval=True)

    def velocity_eval(self, x, t):
        return self.neural_network(x, t) / torch.sqrt(1 - torch.exp(-t))

    def conditional_forward_step(self, x, t1, t2):
        # Not implemented yet return x to avoid error for 2D plotting.
        return x

    def velocity_step(self, x, t1, dt):
        # Not implemented yet return x to avoid error for 2D plotting.
        return x

    def sample_forward(self, x, t_id, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_id]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            t_id
        ]

        return (
            sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def sample_time(self, shape, device):
        bs, dim = shape[0], len(shape)
        t_id = torch.randint(
            0,
            self.nb_time_steps_train,
            (bs,) + (dim - 1) * (1,),
            device=device,
        ).long()
        t = self.times_train[t_id]

        return t, t_id

    def loss(
        self,
        x,
        loss_type="l2",
    ):
        t, t_id = self.sample_time(x.shape, x.device)
        noise = torch.randn_like(x)
        x_noisy = self.sample_forward(x, t_id, noise)

        predicted_noise = self.neural_network(x_noisy, t)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def backward_discrete_time(self):
        return True

    def get_time_from_id(self, t_id, shape, dim, device):
        t_id = torch.full((shape[0],) + (dim - 1) * (1,), t_id).to(device)
        t = self.times_eval[t_id]
        return t

    def velocity_eval(self, x, t):
        return self.neural_network(x, t)

    def velocity_step(self, x, t_id, backward=True):
        if not backward:
            raise NotImplementedError
        # return_velocities = return_neural_net or return_velocities
        (
            times,
            betas,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
        ) = (
            self.times_eval,
            self.betas_eval,
            self.sqrt_one_minus_alphas_cumprod_eval,
            self.sqrt_recip_alphas_eval,
            self.posterior_variance_eval,
        )
        t_id = torch.full((x.shape[0],) + (x.dim() - 1) * (1,), t_id).to(
            x.device
        )
        t = times[t_id]

        # if return_velocities:
        #     return self.neural_network(x, t)

        betas_t = betas[t_id]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t_id]
        sqrt_recip_alphas_t = sqrt_recip_alphas[t_id]

        model_mean = sqrt_recip_alphas_t * (
            x
            - betas_t
            * self.neural_network(x, t)
            / sqrt_one_minus_alphas_cumprod_t
        )

        # Sampling so assuming all t_id are the same
        if t_id[0] == 0:
            return model_mean
        else:
            posterior_variance_t = posterior_variance[t_id]
            noise = torch.randn_like(x)
            return_val = model_mean + torch.sqrt(posterior_variance_t) * noise
            return return_val
