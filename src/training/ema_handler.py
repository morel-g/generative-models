from torch.optim.swa_utils import get_ema_multi_avg_fn


class EMAHandler:
    def __init__(
        self,
        decay: float,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
    ):
        """
        Initializes the EMAHandler with specified parameters for exponential moving average.

        Parameters:
        - decay (float): The (maximum) decay rate for the EMA calculation.
        - min_decay (float): The minimum decay rate allowed for the EMA calculation.
        - update_after_step (int): The number of steps after which EMA updates begin.
        - use_ema_warmup (bool): Whether to use a warmup phase for the EMA calculation.
        - inv_gamma (float): The inverse gamma value used for calculating the decay during warmup.
        - power (float): The power to which the decay calculation is raised.
        """
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.step = 0

    def get_decay(self, optimization_step: int) -> float:
        """
        Calculates the current decay value based on the optimization step.

        Parameters:
        - optimization_step (int): The current optimization step.

        Returns:
        - float: The calculated decay value.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    def ema_multi_avg_fn(
        self, ema_param_list: list, current_param_list: list, num_updates: int
    ) -> None:
        """
        Updates the EMA parameters using the provided current parameters.

        Parameters:
        - ema_param_list (list): The list of EMA parameters to update.
        - current_param_list (list): The list of current model parameters.
        - num_updates (int): The number of updates performed.
        """
        current_decay = self.get_decay(self.step)
        original_ema_fn = get_ema_multi_avg_fn(decay=current_decay)
        original_ema_fn(ema_param_list, current_param_list, num_updates)
        self.step += 1
