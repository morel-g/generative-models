import torch
from src.case import Case


def get_time_scheduler(transition_case: str, nb_time_steps: int) -> torch.Tensor:
    """
    Computes the time scheduler based on training steps.

    Parameters:
        transition_case (str): The transition matrix used during training.
        nb_time_steps (int): The number of time steps to evaluate
        the time scheduler.

    Returns:
    - torch.Tensor: A tensor containing beta values for time scheduling.
    """
    if transition_case == Case.uniform:
        steps = torch.arange(nb_time_steps + 1) / nb_time_steps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = torch.min(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999))
        return betas
    elif transition_case == Case.absorbing:
        return 1.0 / torch.linspace(nb_time_steps, 1.0, steps=nb_time_steps)
    else:
        raise ValueError(f"Unkown transition matrix {transition_case}")


class MarkovHandler(torch.nn.Module):
    def __init__(
        self,
        nb_tokens,
        transition_case,
        nb_time_steps,
        store_transition_matrices: bool = False,
    ):
        super(MarkovHandler, self).__init__()
        self.nb_tokens = nb_tokens
        self.transition_case = transition_case
        self.nb_time_steps = nb_time_steps
        self.store_transition_matrices = store_transition_matrices
        self.time_scheduler = get_time_scheduler(transition_case, self.nb_time_steps)
        if self.store_transition_matrices:
            Qt, Qt_bar = self._construct_transition_matrices(self.time_scheduler)
            self.register_buffer("Qt_bar", Qt_bar)
            self.register_buffer("Qt", Qt)
        else:
            (
                Qt_coef,
                Qt_bar_coef,
            ) = self._construct_transition_coef(self.time_scheduler)

            self.register_buffer("Qt_bar_coef", Qt_bar_coef)
            self.register_buffer("Qt_coef", Qt_coef)
            print(f"Qt_bar_coef.shape {Qt_bar_coef.shape}")

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

    def _construct_transition_matrices(
        self,
        time_scheduler: torch.tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Constructs transition matrices M for the model.
        Given a probability vector p the new probility vector is given by M p

        Parameters:
            time_scheduler (torch.Tensor): A 1-D tensor containing a sequence of
            beta values, each of which corresponds to a time step in the scheduling.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]:
        - A tensor containing stacked matrices Qt_bar.
        - A tensor containing stacked matrices Qt.
        """
        N = self.nb_tokens
        transition_case = self.transition_case
        if transition_case == Case.uniform:
            get_transition_matrix = self._get_uniform_transition_matrix
        elif transition_case == Case.absorbing:
            get_transition_matrix = self._get_absorbing_transition_matrix
        else:
            raise ValueError("Unkown transition case {transition_case}")

        beta_0 = time_scheduler[0]
        Q0 = get_transition_matrix(beta_0, N)
        Qt_bar = [Q0]
        Qt = [Q0]
        for i in range(1, time_scheduler.shape[0]):
            beta_i = time_scheduler[i]
            Qi = get_transition_matrix(beta_i, N)
            Qt.append(Qi)
            Qt_bar.append(torch.mm(Qt_bar[i - 1], Qi))

        return torch.stack(Qt, dim=0), torch.stack(Qt_bar, dim=0)

    def _get_uniform_transition_matrix(self, beta: float, N: int) -> torch.Tensor:
        """
        Computes the uniform transition matrix for q(x_t|x_{t-1}).

        The constructed transition matrix Q is defined as:
        Q_{ij} = beta / N                  if i != j
                1 - \sum_{l \neq i} Q_{il} if i==j.

        Parameters:
        - beta (float): The beta value for matrix computation.
        - N (int): The number of tokens.

        Returns:
        - torch.Tensor: The computed transition matrix.
        """
        mat = torch.full(
            (N, N),
            fill_value=beta / N,
        )

        diag_val = 1.0 - beta * (N - 1.0) / N
        diag_indices = torch.diag(torch.ones(N, dtype=torch.float))
        mat = mat * (1 - diag_indices) + diag_val * diag_indices
        return mat

    def _get_absorbing_transition_matrix(self, beta: float, N: int) -> torch.Tensor:
        """
        Computes the absorbing transition matrix with an abosorbing state at 0.

        The constructed transition matrix Q is defined as:
        Q_{ij} = beta    if j=0
                1 - beta if i==j.

        Parameters:
        - beta (float): The beta value for matrix computation.
        - N (int): The number of tokens.

        Returns:
        - torch.Tensor: The computed transition matrix.
        """
        diagonal = torch.full((N,), 1 - beta)

        # Create a diagonal matrix using the 1D tensor
        matrix = torch.diag(diagonal)
        matrix[0, :] += beta
        return matrix

    def _construct_transition_coef(
        self, time_scheduler: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        if self.transition_case == Case.uniform:
            construct_coef_step = self._construct_uniform_coef_step
        elif self.transition_case == Case.absorbing:
            construct_coef_step = self._construct_absorbing_coef_step
        else:
            raise ValueError(f"Unkown transition matrix {self.transition_case}")

        Qt_coef = []
        Qt_bar_coef = []

        for i in range(time_scheduler.shape[0]):
            beta_i = time_scheduler[i]
            construct_coef_step(Qt_coef, Qt_bar_coef, beta_i)
        Qt_coef, Qt_bar_coef = self._finalize_construct_coef(Qt_coef, Qt_bar_coef)
        return Qt_coef, Qt_bar_coef

    def _construct_uniform_coef_step(self, Qt_coef, Qt_bar_coef, beta):
        def append_coef(coef_list, coef):
            coef_list.append(torch.stack(coef, dim=0))

        append_coef(Qt_coef, [1 - beta, beta])
        if not Qt_bar_coef:
            append_coef(Qt_bar_coef, [1 - beta, beta])
        else:
            prev_coef = Qt_bar_coef[-1]
            append_coef(
                Qt_bar_coef,
                [
                    (1 - beta) * prev_coef[0],
                    beta * prev_coef[0] + prev_coef[1],
                ],
            )

    def _construct_absorbing_coef_step(self, Qt_coef, Qt_bar_coef, beta):
        def append_coef(coef_list, coef):
            coef_list.append(torch.stack(coef, dim=0))

        append_coef(Qt_coef, [1 - beta, beta])
        if not Qt_bar_coef:
            append_coef(Qt_bar_coef, [1 - beta, beta])
        else:
            prev_coef = Qt_bar_coef[-1]
            append_coef(
                Qt_bar_coef,
                [
                    (1 - beta) * prev_coef[0],
                    1 - (1 - beta) * prev_coef[0],
                ],
            )

    def _finalize_construct_coef(self, Qt_coef, Qt_bar_coef):
        Qt_coef, Qt_bar_coef = torch.stack(Qt_coef, dim=0), torch.stack(
            Qt_bar_coef, dim=0
        )
        if self.transition_case == Case.uniform:
            Qt_coef[:, 1] = Qt_coef[:, 1] / self.nb_tokens
            Qt_bar_coef[:, 1] = Qt_bar_coef[:, 1] / self.nb_tokens

        return Qt_coef, Qt_bar_coef

    def extract_rows_Qt(self, t_id: torch.Tensor, x_id: torch.Tensor) -> torch.Tensor:
        """
        Extracts specific rows from Qt using t_id and x_id.

        Parameters:
        - Qts (torch.Tensor): Input tensor from which rows are to be extracted.
        - t_id (torch.Tensor): Tensor index for the time dimension.
        - x_id (torch.Tensor): Tensor index for the x dimension.

        Returns:
        - torch.Tensor: Extracted rows from the input tensor Qts.
        """
        if self.store_transition_matrices:
            if t_id.dim() != x_id.dim():
                t_id_broadcast = self._broadcast(t_id, x_id.dim())
            else:
                t_id_broadcast = t_id
            return self.Qt[t_id_broadcast, x_id]
        else:
            return self._extract_rows_Qt_from_coef(t_id, x_id)

    def extract_cols_Qt_bar(
        self, t_id: torch.Tensor, x_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts specific columns from Qt_bar using t_id and x_id.

        Parameters:
        - t_id (torch.Tensor): Tensor index for the time dimension.
        - x_id (torch.Tensor): Tensor index for the x dimension.

        Returns:
        - torch.Tensor: Extracted columns from the input tensor Qts.
        """
        if self.store_transition_matrices:
            if t_id.dim() != x_id.dim():
                t_id_broadcast = self._broadcast(t_id, x_id.dim())
            else:
                t_id_broadcast = t_id

            return self.Qt_bar[t_id_broadcast, ..., x_id]
        else:
            return self._extract_cols_Qt_bar_from_coef(t_id, x_id)

    def Qt_x(self, t_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix multiplication of Qt  evaluated at t_id and x with:
        x.shape = (bs, seq_len, nb_tokens).

        Parameters:
        - t_id (Tensor): A tensor representing transition IDs.
        - x (Tensor): A matrix of shape (bs, seq_len, nb_tokens).

        Returns:
        - Tensor: The result of the matrix multiplication.
        """
        if self.store_transition_matrices:
            Q = self.Qt[t_id]
            return torch.matmul(Q.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
        else:
            t_id = t_id.squeeze()
            id_coef, a_coef = (
                self.Qt_coef[t_id][:, 0],
                self.Qt_coef[t_id][:, 1],
            )
            return self._product_mat_vect_from_coef(id_coef, a_coef, x)

    def Qt_bar_x(self, t_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix multiplication of Qt_bar evaluated at t_id
        and x with: x.shape = (bs, seq_len, nb_tokens).

        Parameters:
        - t_id (Tensor): A tensor representing transition IDs.
        - x (Tensor): A matrix of shape (bs, seq_len, nb_tokens).

        Returns:
        - Tensor: The result of the matrix multiplication.
        """
        if self.store_transition_matrices:
            Q = self.Qt_bar[t_id]
            return torch.matmul(Q.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
        else:
            t_id = t_id.squeeze()
            id_coef, a_coef = (
                self.Qt_bar_coef[t_id][:, 0],
                self.Qt_bar_coef[t_id][:, 1],
            )
            return self._product_mat_vect_from_coef(id_coef, a_coef, x)

    def _product_mat_vect_from_coef(
        self,
        id_coef: torch.Tensor,
        a_coef: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the matrix multiplication of the transition matrix M and x.

        Parameters:
            - id_coef (Tensor): Coefficients for the identity matrix part.
            - a_coef (Tensor): Additional coefficients for the matrix.
            - x (Tensor): A matrix of shape (bs, seq_len, nb_tokens).

        Returns:
            - Tensor: The result of the matrix multiplication.
        """
        id_coef = self._broadcast(id_coef, x.dim())
        a_coef = self._broadcast(a_coef, x.dim())

        if self.transition_case == Case.uniform:
            # M = id_coef*Id + a_coef * 1 1^T
            return id_coef * x + (a_coef * x).sum(dim=-1, keepdim=True)
            #
        elif self.transition_case == Case.absorbing:
            # M = id_coef*Id + a_coef * e_0 1^T
            mask = torch.zeros_like(x)
            mask[..., 0] = 1  # Assuming the mask id is 0.
            return id_coef * x + mask * (a_coef * x).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unkown transition case {self.transition_case}")

    def _extract_rows_Qt_from_coef(
        self, t_id: torch.Tensor, x_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract rows (or equivalently a columns) given by x_id of the matrix M
        defined by M = (a_coef/nb_tokens) * 1 1^T + id_coef*Id

        Parameters:
        - t_id (torch.Tensor): The time index to extract the row from.
        - a_coef (torch.Tensor): The first tensor of coefficients for the matrix.
        - x_id (torch.Tensor): The indices of the vector (row or column) to extract.

        Returns:
        - torch.Tensor: The extracted vector from the matrix.
        """
        t_id = t_id.squeeze()
        id_coef, a_coef = (
            self.Qt_coef[t_id][:, 0],
            self.Qt_coef[t_id][:, 1],
        )
        target_dim = x_id.dim() + 1
        id_coef = self._broadcast(id_coef, target_dim)
        a_coef = self._broadcast(a_coef, target_dim)

        # Create a mask to select the elements to be updated
        id_mask = torch.arange(self.nb_tokens, device=x_id.device).view(
            1, 1, -1
        ) == x_id.unsqueeze(-1)

        if self.transition_case == Case.uniform:
            return id_coef * id_mask.float() + a_coef  # broadcasted
        elif self.transition_case == Case.absorbing:
            absorbing_mask = torch.zeros(
                (1, 1, self.nb_tokens), device=x_id.device
            ) == x_id.unsqueeze(-1)

            return id_coef * id_mask.float() + a_coef * absorbing_mask.float()
        else:
            raise ValueError(f"Unkown transition case {self.transition_case}")

    def _extract_cols_Qt_bar_from_coef(
        self, t_id: torch.Tensor, x_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract rows (or equivalently a columns) given by x_id of the matrix M
        defined by M = (a_coef/nb_tokens) * 1 1^T + id_coef*Id

        Parameters:
        - t_id (torch.Tensor): The time index to extract the row from.
        - x_id (torch.Tensor): The indices of the vector (row or column) to extract.

        Returns:
        - torch.Tensor: The extracted vector from the matrix.
        """
        t_id = t_id.squeeze()

        id_coef, a_coef = (
            self.Qt_bar_coef[t_id][..., 0],
            self.Qt_bar_coef[t_id][..., 1],
        )
        target_dim = x_id.dim() + 1
        id_coef = self._broadcast(id_coef, target_dim)

        a_coef = self._broadcast(a_coef, target_dim)

        # Create a mask to select the elements to be updated
        id_mask = torch.arange(self.nb_tokens, device=x_id.device).view(
            1, 1, -1
        ) == x_id.unsqueeze(-1)

        if self.transition_case == Case.uniform:
            return id_coef * id_mask.float() + a_coef  # broadcasted
        elif self.transition_case == Case.absorbing:
            absorbing_part = torch.zeros(
                (x_id.shape[0], 1, self.nb_tokens), device=x_id.device
            )
            absorbing_part[..., 0] = a_coef.squeeze(-1)

            return id_coef * id_mask.float() + absorbing_part
        else:
            raise ValueError(f"Unkown transition case {self.transition_case}")
