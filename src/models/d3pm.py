import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from src.models.model import Model
from src.models.helpers.markov_handler import MarkovHandler
from src.case import Case


class D3PM(Model):
    """
    D3PM model is a discrete diffusion model.
    """

    def __init__(
        self,
        data_type: str,
        model_params: dict,
        seq_length: int,
        nb_time_steps_eval: int,
        nb_time_steps_train: int = None,
        T_final: float = 1.0,
        beta_case: str = Case.constant,
        lambda_hybrid_loss: float = 0.0,
        transition_case: str = Case.uniform,
        store_transition_matrices: bool = False,
    ) -> None:
        """
        Initializes the D3PM class.

        Parameters:
        - data_type (str): Type of data the model will handle.
        - model_params (dict): Parameters related to the model configuration.
        - seq_length (int): Length of the sequence in the model.
        - nb_time_steps_eval (int): Number of evaluation time steps.
        - nb_time_steps_train (int, optional): Number of training time steps.
        - T_final (float, optional): Final time point. Defaults to 1.0.
        - beta_case (str, optional): The case for beta. Defaults to `Case.constant`.
        - lambda_hybrid_loss (float, optional): Coeficient for hybrid loss
        (add ponderation in the loss for the initial time).
        - transition_case (str, optional): the choice for the type of transition
        matrix.
        - store_transition_matrices (bool, optional): whether to store the
        transition matrices (can be memory consuming if the number of tokens
        is large).

        Returns:
        - None
        """
        super(D3PM, self).__init__(
            data_type,
            model_params,
            nb_time_steps_eval,
            nb_time_steps_train,
            T_final=T_final,
        )

        self.beta_case = beta_case
        self.nb_tokens = model_params["nb_tokens"]
        self.seq_length = seq_length
        self.eps = 1e-6
        self.lambda_hybrid_loss = lambda_hybrid_loss
        self.store_transition_matrices = store_transition_matrices
        self.transition_case = transition_case
        self.markov_handler = MarkovHandler(
            self.nb_tokens,
            transition_case,
            nb_time_steps_train,
            store_transition_matrices,
        )

        if self.store_transition_matrices:
            self.register_buffer("Qt_bar", self.markov_handler.Qt_bar)
            self.register_buffer("Qt", self.markov_handler.Qt)
        else:
            self.register_buffer("Qt_bar_coef", self.markov_handler.Qt_bar_coef)
            self.register_buffer("Qt_coef", self.markov_handler.Qt_coef)

    def sample_with_discrete_time(self) -> bool:
        """
        Determines if the time sampling is discrete.

        Returns:
        - bool: True if the time sampling is discrete, otherwise False.
        """
        return True

    def sample_prior_x(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Samples prior values for x given a specified shape and device.

        Parameters:
        - shape (torch.Size): Desired shape of the output tensor.
        - device (torch.device): The device on which the tensor will reside.

        Returns:
        - torch.Tensor: A tensor filled with random integers.
        """
        if self.transition_case == Case.uniform:
            return torch.randint(0, self.nb_tokens, shape, device=device)
        elif self.transition_case == Case.absorbing:
            return torch.zeros(shape, device=device, dtype=torch.int)
        else:
            raise ValueError(f"Unkown transition case {self.transition_case}")

    def _broadcast(self, t: torch.Tensor, n_dim: int) -> torch.Tensor:
        """
        Broadcasts the tensor t to a specified number of dimensions.

        Parameters:
        - t (torch.Tensor): Input tensor to be broadcasted.
        - n_dim (int): The number of dimensions to broadcast to.

        Returns:
        - torch.Tensor: Broadcasted tensor.
        """
        return t.view(*[t.shape[0]] + [1] * (n_dim - 1))

    # q_posterior_logits
    def predict_given_xt_and_x0(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t_id: torch.Tensor,
        is_x0_int: bool = True,
    ) -> torch.Tensor:
        """
        Compute logits of q(x_{t-1} | x_t, x_start).

        Parameters:
        - xt (torch.Tensor): Tensor representing x_t.
        - x0 (torch.Tensor): Tensor representing x_start.
        - t_id (torch.Tensor): Tensor of times IDs.
        - is_x0_int (bool): Flag indicating if x0 is an integer.

        Returns:
        - torch.Tensor: The computed logits.
        """
        t_id = t_id.squeeze()
        factor_1 = self.markov_handler.extract_rows_Qt(t_id, xt)

        if is_x0_int:
            factor_2 = self.markov_handler.extract_cols_Qt_bar(t_id - 1, x0)
            x_start = torch.nn.functional.one_hot(
                x0.long(), num_classes=self.nb_tokens
            ).float()
        else:
            factor_2 = self.markov_handler.Qt_bar_x(t_id - 1, x0)
            x_start = x0

        t_id_broadcast = self._broadcast(t_id, factor_1.dim())

        out = torch.log(factor_1 + self.eps) + torch.log(factor_2 + self.eps)
        return torch.where(t_id_broadcast == 0, torch.log(x_start + self.eps), out)

    # p_logits
    def predict_given_xt(
        self, xt: torch.Tensor, t_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits of p(x_{t-1} | x_t).

        Parameters:
        - xt (torch.Tensor): Tensor representing x_t.
        - t_id (torch.Tensor): Tensor of times IDs.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the predicted logits
        for x_{t-1} and x_start.
        """
        t = self.times_train[t_id]
        x0_pred = self.neural_network(xt, t)
        t_broadcast = self._broadcast(t_id, x0_pred.dim())
        xt_minus_1_pred = torch.where(
            t_broadcast == 0,
            F.log_softmax(x0_pred, dim=-1),
            self.predict_given_xt_and_x0(
                xt, F.softmax(x0_pred, dim=-1), t_id, is_x0_int=False
            ),
        )

        return xt_minus_1_pred, x0_pred

    def _sample_from_categorical(
        self, x: torch.Tensor, noise_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from a categorical distribution using the Gumbel-Softmax trick.

        Parameters:
        - x (torch.Tensor): Logits of the categorical distribution.
        - noise_mask (Optional[torch.Tensor]): A mask for the noise, if any.

        Returns:
        - torch.Tensor: Sampled indices from the categorical distribution.
        """
        if noise_mask is None:
            noise_mask = torch.ones_like(x)
        noise = torch.rand_like(x)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(x + noise_mask * gumbel_noise, axis=-1)

    def sample_from_int(
        self, x_start_id: torch.Tensor, t_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_start) by adding noise to the data.

        Parameters:
        - x_start_id (torch.Tensor): Tensor of starting indices.
        - t_id (torch.Tensor): Tensor of times IDs.

        Returns:
        - torch.Tensor: Sampled tensor after adding noise.
        """
        x = torch.log(
            self.markov_handler.extract_cols_Qt_bar(t_id, x_start_id) + self.eps
        )
        x = self._sample_from_categorical(x)

        return x

    def _sample_time(self, x_shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Sample time values based on the shape of x.

        Parameters:
        - x_shape (tuple): Shape of the input tensor x.
        - device (torch.device): Device to allocate tensor on.

        Returns:
        - torch.Tensor: Tensor with sampled time values.
        """
        dim = len(x_shape)
        t_shape = (x_shape[0],) + (dim - 1) * (1,)
        t_id = torch.randint(0, self.nb_time_steps_train, t_shape, device=device)

        return self.times_train[t_id], t_id

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Computed loss value.
        """
        x0_id = x
        t, t_id = self._sample_time(x.shape, x.device)
        xt = self.sample_from_int(x0_id, t_id)
        # Log probability of eq (3) of the paper
        log_xt_minus_1 = self.predict_given_xt_and_x0(xt, x0_id, t_id, is_x0_int=True)
        log_xt_minus_1_pred, x0_pred = self.predict_given_xt(xt, t_id)
        kl_loss_xt = self._compute_kl_loss(
            F.log_softmax(log_xt_minus_1_pred, dim=-1),
            F.log_softmax(log_xt_minus_1, dim=-1),
            log_target=True,
        )
        loss = kl_loss_xt
        if self.lambda_hybrid_loss > 0:
            kl_loss_x0 = self._compute_kl_loss(
                F.log_softmax(x0_pred + self.eps, dim=-1),
                torch.nn.functional.one_hot(
                    x0_id.long(), num_classes=self.nb_tokens
                ).float(),
                log_target=False,
            )
            loss += self.lambda_hybrid_loss * kl_loss_x0

        return loss.mean()

    def _compute_kl_loss(self, x_pred, x_true, log_target=False):
        kl_loss = F.kl_div(
            x_pred,
            x_true,
            reduction="none",
            log_target=log_target,
        ).sum(-1)
        kl_loss = torch.mean(kl_loss, dim=tuple(range(1, kl_loss.dim())))
        return kl_loss

    def conditional_forward_step(self, x, t_id):
        if t_id == self.nb_time_steps_eval - 1:
            pass
        t_id = torch.full((x.shape[0],), t_id)
        x_one_hot = torch.nn.functional.one_hot(
            x.long(), num_classes=self.nb_tokens
        ).float()

        log_prob = torch.log(self.markov_handler.Qt_x(t_id, x_one_hot) + self.eps)
        return self._sample_from_categorical(log_prob)

    def velocity_step(self, x, t_id, backward=False):
        if backward:
            xt_minus_1, _ = self.predict_given_xt(x, t_id)
            nonzero_mask = (
                (t_id != 0)
                .to(dtype=xt_minus_1.dtype)
                .reshape(xt_minus_1.shape[0], *([1] * (len(xt_minus_1.shape) - 1)))
            )

            return self._sample_from_categorical(xt_minus_1, nonzero_mask)

        else:
            # Not implemented yet
            return torch.zeros_like(x)
