from .model import Model

from diffusers import DDPMScheduler
import torch
import torch.nn.functional as F


def scheduler_conditional_step(scheduler, x, timestep, noise=None):
    noise = torch.randn_like(x) if noise is None else noise
    alphas = scheduler.alphas
    sqrt_alphas = alphas**0.5
    sqrt_betas = (1 - alphas) ** 0.5
    return sqrt_alphas[timestep] * x + sqrt_betas * noise


class DenoiseModel(Model):
    def __init__(
        self,
        data_type,
        model_params,
        nb_time_steps,
        nb_time_steps_eval,
        T_final=200,
    ):
        super(DenoiseModel, self).__init__(
            data_type,
            model_params,
            nb_time_steps,
            nb_time_steps_eval,
            T_final=T_final,
        )
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=nb_time_steps)

    def conditional_forward_step(self, x, t1, t2):
        # Not implemented yet return x to avoid error when 2D plotting.
        return x

    def velocity_step(self, x, t1, dt):
        # Not implemented yet return x to avoid error when 2D plotting.
        return x

    def loss(self, x, t_id_val=None):
        # Sample a random timestep for each image
        noise = torch.randn(x.shape).to(x.device)
        bs, dim = x.shape[0], len(x.shape)
        timesteps = (
            torch.randint(
                0,
                self.noise_scheduler.num_train_timesteps,
                (bs,) + (dim - 1) * (1,),
            )
            .long()
            .to(x.device)
        )
        noisy_images = self.noise_scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.neural_network(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def backward_step(x, t_id):
        noisy_residual = self.model(x, t_id).sample
        previous_noisy_sample = scheduler.step(
            noisy_residual, t, input
        ).prev_sample
        input = previous_noisy_sample
        return self.noise_scheduler.step(t_id)
