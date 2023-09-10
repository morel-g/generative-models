# https://huggingface.co/blog/annotated-diffusion
import torch
from tqdm.auto import tqdm
import numpy as np
from ...utils import t_from_id
from ...case import Case


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(
    x_start,
    t_id,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    noise=None,
):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t_id]
    # extract(sqrt_alphas_cumprod, t_id, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t_id]
    # extract(sqrt_one_minus_alphas_cumprod, t_id, x_start.shape)

    return (
        sqrt_alphas_cumprod_t * x_start
        + sqrt_one_minus_alphas_cumprod_t * noise
    )


@torch.no_grad()
def p_sample(
    model,
    x,
    t_id,
    times,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    return_velocities=False,
):
    t_id = torch.full((x.shape[0],) + (x.dim() - 1) * (1,), t_id).to(x.device)
    t = times[t_id]  # .repeat(
    #         (x.shape[0],) + (x.dim() - 1) * (1,)
    #     )
    if return_velocities:
        return model(x, t)

    # t_id = torch.full((x.shape[0],), t_id).to(x.device)
    betas_t = betas[t_id]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t_id]
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_id]

    # Equation 11 in the paper

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    # Sampling so assuming all t_id are the same
    if t_id[0] == 0:
        return model_mean
    else:
        posterior_variance_t = posterior_variance[t_id]
        # extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:

        return_val = model_mean + torch.sqrt(posterior_variance_t) * noise
        return return_val


@torch.no_grad()
def p_sample_loop(
    model,
    shape,
    nb_time_steps,
    dt,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    return_trajectories=False,
    # return_velocities=False,
):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    x = torch.randn(shape, device=device)
    x_traj = [x.cpu().numpy()]

    for i in tqdm(
        reversed(range(0, nb_time_steps)),
        desc="sampling loop time step",
        total=nb_time_steps,
    ):
        x = p_sample(
            model,
            x,
            i,
            dt,
            betas,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
        )

        if return_trajectories:
            x_traj.append(x.cpu().numpy())

    return x if not return_trajectories else np.array(x_traj)
    # return return_val if not return_velocities else (return_val, np.array(scores))


@torch.no_grad()
def sample(
    model,
    image_size,
    timesteps,
    dt,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    batch_size=16,
    channels=3,
):
    return p_sample_loop(
        model,
        (batch_size, channels, image_size, image_size),
        timesteps,
        dt,
        betas,
        sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas,
        posterior_variance,
    )
