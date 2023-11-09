import sys

sys.path.append("..")
import unittest
import torch
import numpy as np
from scipy.integrate import quad
from src.models.helpers.adapt_dt import (
    exact_adapt_dt_constant_pde_coef,
    adapt_dt_pdf,
)


class TestVectorField(unittest.TestCase):
    def test_exact_constant_pde_coef(self):
        def sigma(t, gamma):
            return (1 - np.exp(-2 * gamma * t)) ** 0.5

        def integral_inverse_sigma(t1, t2, gamma):
            v1 = (1 - sigma(t1, gamma)) / (1 + sigma(t1, gamma))
            v2 = (1 + sigma(t2, gamma)) / (1 - sigma(t2, gamma))
            return (1.0 / gamma) * np.log(v1 * v2)

        def params_test(nb_time_steps, T_final, gamma):
            dt = exact_adapt_dt_constant_pde_coef(
                nb_time_steps, T_final, gamma=gamma
            )
            times = np.zeros(nb_time_steps + 1)
            times[1:] = torch.cumsum(dt, dim=0).detach().cpu().numpy()

            first_int = integral_inverse_sigma(times[0], times[1], gamma)
            for i in range(1, len(times) - 1):
                self.assertAlmostEqual(
                    first_int,
                    integral_inverse_sigma(times[i], times[i + 1], gamma),
                    delta=1e-4,
                )

        params_test(50, 3.0, 0.5)
        params_test(50, 3.0, 1.0)

    def test_adapt_dt_pdf(self):
        def sigma_eval(t):
            t_new = 0.1 * t / 2.0 + 0.25 * 19.9 * t**2
            return np.sqrt(1 - np.exp(-2 * t_new))

        func = lambda t: 1.0 / sigma_eval(t)
        nb_time_steps = 50
        T_final = 3.0
        dt, times = adapt_dt_pdf(func, nb_time_steps, 0.0, T_final)

        first_int = quad(func, times[0], times[1])[0]
        for i in range(1, len(times) - 1):
            self.assertAlmostEqual(
                first_int,
                quad(func, times[i], times[i + 1])[0],
                delta=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
