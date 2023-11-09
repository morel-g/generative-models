import torch
import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton


def exact_adapt_dt_constant_pde_coef(nb_time_steps, T_final, gamma=1.0):
    times = torch.zeros(nb_time_steps + 1)

    # Constant exp
    # times[0] = 1.0
    # for i in range(1, nb_time_steps):
    #     times[i] = ((np.exp(-T_final) - 1.0) / nb_time_steps) + times[i - 1]
    # times = -torch.log(times)

    # Constant sigma
    # times[0] = 0.0
    # C = -np.sqrt(1 - np.exp(-2 * T_final)) / nb_time_steps
    # for i in range(1, nb_time_steps):
    #     times[i] = times[i - 1] - C
    # times = -0.5 * torch.log(1 - times**2)

    # Same as denoise model
    # dt = 2 * T_final / ((nb_time_steps + 1) * (nb_time_steps + 2))
    # times[0] = 0.0
    # for i in range(1,nb_time_steps + 1):
    #     times[i] = times[i-1] + i*dt

    # For L1 loss and 1/sigma(t) as pdf
    sigma_T = np.sqrt(1 - np.exp(-2 * gamma * T_final))
    C = (
        (1.0 / (2 * gamma))
        * np.log((1 + sigma_T) / (1 - sigma_T))
        / nb_time_steps
    )
    times[0] = 0.0
    for i in range(1, nb_time_steps + 1):
        times[i] = C + times[i - 1]

    times[1:] = -(1.0 / (2 * gamma)) * np.log(
        1 - (1 - 2.0 / (np.exp(2 * gamma * times[1:]) + 1)) ** 2
    )

    times[0], times[-1] = 0.0, T_final
    return times[1:] - times[:-1]


def adapt_dt_pdf_start_right(f_torch, nb_time_steps, t0, t1, x0=None):
    def func(t):
        with torch.no_grad():
            return f_torch(torch.tensor(t, dtype=torch.double)).item()

    times = np.zeros(nb_time_steps + 1) + t0
    times[-1] = t1
    quadrature = quad(func, t0, t1)
    slice_time_int = quadrature[0] / nb_time_steps

    for i in reversed(range(2, nb_time_steps + 1)):
        g = lambda x: quad(func, x, times[i])[0] - slice_time_int
        g_prime = lambda x: -func(x)
        if i == nb_time_steps:
            x0 = times[i].item() if x0 is None else x0
        else:
            if t0 < t1:
                x0 = (
                    times[i]
                    - min(times[i + 1] - times[i], times[i] - times[i - 1])
                    / 2.0
                ).item()
            else:
                x0 = (
                    times[i]
                    + min(times[i] - times[i + 1], times[i - 1] - times[i])
                    / 2.0
                ).item()
        times[i - 1] = newton(g, x0, g_prime)

    times[0], times[-1] = t0, t1
    # eps_num = 2 * np.finfo(np.float32).eps
    return torch.tensor(times[1:] - times[:-1]).type(
        torch.float32
    ), torch.tensor(times).type(
        torch.float32
    )  # + eps_num


def adapt_dt_pdf(f_torch, nb_time_steps, t0, t1, start_left=False, x0=None):
    if not start_left:
        return adapt_dt_pdf_start_right(f_torch, nb_time_steps, t0, t1, x0=x0)
    else:
        dt, times = adapt_dt_pdf_start_right(
            f_torch, nb_time_steps, t1, t0, x0=x0
        )
        return torch.flip(-dt, (0,)), torch.flip(times, (0,))


def integrate(f_torch, t0, t1):
    def func(t):
        with torch.no_grad():
            return f_torch(torch.tensor(t, dtype=torch.double)).item()

    return quad(func, t0, t1)[0]
