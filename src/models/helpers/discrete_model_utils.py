import torch
from src.case import Case


def get_time_scheduler(
    transition_case: str, nb_time_steps: int
) -> torch.Tensor:
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
        betas = torch.min(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999)
        )
        return betas
    elif transition_case == Case.absorbing:
        return 1.0 / torch.linspace(nb_time_steps, 1.0, steps=nb_time_steps)
    else:
        raise ValueError(f"Unkown transition matrix {transition_case}")


def construct_transition_matrices(
    nb_tokens: int,
    time_scheduler: torch.tensor,
    transition_case: str = Case.uniform,
) -> (torch.Tensor, torch.Tensor):
    """
    Constructs transition matrices M for the model.
    Given a probability vector p the new probility vector is given by M p

    Parameters:
        nb_tokens (int): The number of tokens in the sequence for which the
        transition matrices are being constructed.
        time_scheduler (torch.Tensor): A 1-D tensor containing a sequence of
        beta values, each of which corresponds to a time step in the scheduling.
        transition_case (str): The choice for the transition matrix.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]:
    - A tensor containing stacked matrices Qt_bar.
    - A tensor containing stacked matrices Qt.
    """
    if transition_case == Case.uniform:
        get_transition_matrix = get_uniform_transition_matrix
    elif transition_case == Case.absorbing:
        get_transition_matrix = get_absorbing_transition_matrix
    else:
        raise ValueError("Unkown transition case {transition_case}")

    N = nb_tokens
    beta_0 = time_scheduler[0]
    Q0 = get_transition_matrix(beta_0, N)
    Qt_bar = [Q0]
    Qt = [Q0]
    for i in range(1, time_scheduler.shape[0]):
        beta_i = time_scheduler[i]
        Qi = get_transition_matrix(beta_i, N)
        Qt.append(Qi)
        Qt_bar.append(torch.mm(Qt_bar[i - 1], Qi))

    return torch.stack(Qt_bar, dim=0), torch.stack(Qt, dim=0)


def get_uniform_transition_matrix(beta: float, N: int) -> torch.Tensor:
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


def get_absorbing_transition_matrix(beta: float, N: int) -> torch.Tensor:
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


# def get_transition_matrix(
#     beta: float, N: int, transition_bands: int
# ) -> torch.Tensor:
#     """
#     Compute a transition matrix.

#     Parameters:
#     - beta (float): The beta value used for calculations.
#     - N (int): Dimension of the square matrix.
#     - transition_bands (int): Number of bands for transition.

#     Returns:
#     - torch.Tensor: The computed transition matrix.
#     """

#     mat = torch.zeros((N, N), dtype=torch.float)
#     off_diag = torch.full(
#         size=(N - 1,),
#         fill_value=beta / float(N),
#         dtype=torch.float,
#     )

#     for k in range(1, transition_bands + 1):
#         mat += torch.diag(off_diag, diagonal=k)
#         mat += torch.diag(off_diag, diagonal=-k)
#         off_diag = off_diag[:-1]

#     # Add diagonal values such that rows sum to one.
#     diag = 1.0 - torch.sum(mat, dim=1)
#     mat += torch.diag(diag)
#     return mat


def construct_transition_coef(
    nb_tokens: int, time_scheduler: torch.tensor
) -> (torch.Tensor, torch.Tensor):
    def append_coef(coef_list, coef):
        coef_list.append(torch.stack(coef, dim=0))

    N = nb_tokens
    beta_0 = time_scheduler[0]
    Qt_coef = []
    Qt_bar_coef = []
    append_coef(Qt_coef, [1 - beta_0, beta_0])
    append_coef(Qt_bar_coef, [1 - beta_0, beta_0])
    for i in range(1, time_scheduler.shape[0]):
        beta_i = time_scheduler[i]
        append_coef(Qt_coef, [1 - beta_i, beta_i])
        prev_coef = Qt_bar_coef[i - 1]
        append_coef(
            Qt_bar_coef,
            [
                (1 - beta_i) * prev_coef[0],
                beta_i * prev_coef[0] + prev_coef[1],
            ],
        )

    return torch.stack(Qt_bar_coef, dim=0), torch.stack(Qt_coef, dim=0)


def product_mat_vect_from_coef(
    transition_case: str,
    id_coef: torch.Tensor,
    a_coef: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the matrix multiplication of the transition matrix M and x.

    Parameters:
        - transition_case (str): The transition case.
        - id_coef (Tensor): Coefficients for the identity matrix part.
        - a_coef (Tensor): Coefficients for the matrix of ones part.
        - x (Tensor): A matrix of shape (bs, seq_len, nb_tokens).

    Returns:
        - Tensor: The result of the matrix multiplication.
    """

    if transition_case == Case.uniform:
        # M = id_coef*Id + (a_coef/nb_tokens) * 1 1^T
        return id_coef * x + (a_coef * x).sum(dim=-1, keepdim=True)
        #
    elif transition_case == Case.absorbing:
        # M = id_coef*Id + (a_coef/nb_tokens) * e_0 1^T
        mask = torch.zeros_like(x)
        mask[:, 0] = 1  # Assuming the mask id is 0.
        return id_coef * x + mask * (a_coef * x).sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unkown transition case {transition_case}")


def extract_vector_from_coef(
    self, id_coef: torch.Tensor, a_coef: torch.Tensor, x_id: torch.Tensor
) -> torch.Tensor:
    """
    Extract rows (or equivalently a columns) given by x_id of the matrix M
    defined by M = (a_coef/nb_tokens) * 1 1^T + id_coef*Id

    Parameters:
    - id_coef (torch.Tensor): The identity coefficients for the matrix.
    - a_coef (torch.Tensor): The first tensor of coefficients for the matrix.
    - x_id (torch.Tensor): The indices of the vector (row or column) to extract.

    Returns:
    - torch.Tensor: The extracted vector from the matrix.
    """
    target_dim = x_id.dim() + 1
    id_coef = self._broadcast(id_coef, target_dim)
    a_coef = self._broadcast(a_coef / self.nb_tokens, target_dim)
    full_shape = x_id.shape + (self.nb_tokens,)
    a_broadcasted = torch.broadcast_to(a_coef, full_shape)

    # Create a mask to select the elements to be updated
    mask = torch.arange(self.nb_tokens, device=x_id.device).view(
        1, 1, -1
    ) == x_id.unsqueeze(-1)
    return a_broadcasted + id_coef * mask.float()
