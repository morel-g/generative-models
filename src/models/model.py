import torch
import numpy as np

from src.case import Case
from src.data_manager.data_type import toy_data_type
from src.neural_networks.neural_network import NeuralNetwork
from src.models.model_utils import (
    reduce_length_array,
    append_trajectories,
    trajectories_to_array,
)
from typing import Optional, Tuple, Union, Any, List


class Model(torch.nn.Module):
    """
    Base class for the models.
    """

    def __init__(
        self,
        data_type: str,
        model_params: dict,
        nb_time_steps_eval: int,
        nb_time_steps_train: Optional[int] = None,
        T_final: float = 1.0,
        T_init: float = 0.0,
        adapt_dt: bool = False,
        img_model_case: str = Case.u_net,
        use_neural_net: bool = True,
    ) -> None:
        """
        Initializes the Model class.

        Parameters:
        - data_type (str): Specifies the type of data the model will handle.
        - model_params (dict): Parameters for the neural network model.
        - nb_time_steps_eval (int): Number of time steps for evaluation.
        - nb_time_steps_train (Optional[int]): Number of time steps for training.
        - T_final (float): The final time for the model simulation.
        - T_init (float): The initial time for the model simulation.
        - adapt_dt (bool): Whether to adapt the time step or not.
        - img_model_case (Case): Specifies the case of the image model.
        - use_neural_net (bool): Whether to use a neural network for the model.

        Returns:
        - None
        """
        super(Model, self).__init__()
        self.register_buffers()
        self.data_type = data_type
        self.model_params = model_params

        self.T_final = T_final
        self.T_init = T_init
        self.adapt_dt = adapt_dt
        self.backward_scheme = Case.euler_explicit
        self.set_nb_time_steps(nb_time_steps_eval, eval=True)
        if nb_time_steps_train is not None:
            self.set_nb_time_steps(nb_time_steps_train, eval=False)
        else:
            self.nb_time_steps_train = None

        if use_neural_net:
            model_case = (
                Case.vector_field
                if data_type in toy_data_type
                else img_model_case
            )
            self.process_params(model_case, model_params)
            self.neural_network = NeuralNetwork(model_case, model_params)
        else:
            self.neural_network = None
        self.return_all_trajs = False

    def register_buffers(self):
        self.register_buffer("dt_train", None)
        self.register_buffer("times_train", None)
        self.register_buffer("dt_eval", None)
        self.register_buffer("times_eval", None)

    @staticmethod
    def _get_noise_like(size_like: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(size_like)

    def get_nb_time_steps_eval(self):
        return self.nb_time_steps_eval

    def compute_uniform_times(
        self, nb_time_steps: int, t0: float, t1: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes uniform time steps between t0 and t1 for nb_time_steps intervals.

        Parameters:
        - nb_time_steps (int): Number of time steps.
        - t0 (float): Initial time.
        - t1 (float): Final time.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: dt and times tensors.
        """

        dt = (t1 - t0) / nb_time_steps
        dt = torch.full((nb_time_steps,), dt)
        times = torch.zeros(nb_time_steps + 1) + t0
        times[1:] = torch.cumsum(dt, dim=0)
        return dt, times

    def set_nb_time_steps(
        self, nb_time_steps: int, eval: bool = False
    ) -> None:
        """
        Sets the number of time steps for the model, either for training or evaluation.

        Parameters:
        - nb_time_steps (int): The number of time steps to set.
        - eval (bool): Whether to set the time steps for evaluation mode.
                        If False, sets for training mode.

        Returns:
        - None
        """
        if eval:
            self.nb_time_steps_eval = nb_time_steps
            self.dt_eval, self.times_eval = self.compute_uniform_times(
                nb_time_steps, self.T_init, self.T_final
            )
        else:
            self.nb_time_steps_train = nb_time_steps
            self.dt_train, self.times_train = self.compute_uniform_times(
                nb_time_steps, self.T_init, self.T_final
            )

    def set_adapt_dt(self, adapt_dt: bool) -> None:
        """
        Sets the adapt_dt flag, which determines whether the time step should be adaptive.

        Parameters:
        - adapt_dt (bool): The value to set for the adapt_dt flag.
                        If True, the time step will be adaptive.

        Returns:
        - None
        """
        self.adapt_dt = adapt_dt
        if self.nb_time_steps_train is not None:
            self.set_nb_time_steps(self.nb_time_steps_train, eval=False)
        self.set_nb_time_steps(self.nb_time_steps_eval, eval=True)

    def get_backward_schemes(self) -> None:
        return [self.backward_scheme]

    def get_backward_scheme(self) -> None:
        return self.backward_scheme

    def set_backward_scheme(self, scheme: str) -> None:
        self.backward_scheme = scheme

    def sample_time(
        self, shape: Tuple[int, ...], device: str, get_id: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Samples time values uniformly between the initial and final times of the model.

        Parameters:
        - shape (Tuple[int, ...]): The shape of the output tensor.
        - device (str): The device on which to create the tensor (e.g., "cpu" or "cuda").
        - get_id (bool): Whether to return a unique identifier for the sampled time.

        Returns:
        - torch.Tensor: The sampled time values. If get_id is True, returns a tuple
        containing the time values and their unique identifier.
        """
        dim = len(shape)

        t = (
            torch.rand((shape[0],) + (dim - 1) * (1,), device=device)
            * (self.T_final - self.T_init)
            + self.T_init
        )
        t_id = None

        return t if not get_id else (t, t_id)

    def velocity_eval(self, x, t):
        """Must be implemented in subclass."""
        raise NotImplementedError(
            "This method should be overridden by subclass."
        )

    def loss(self, x):
        """Must be implemented in subclass."""
        raise NotImplementedError(
            "This method should be overridden by subclass."
        )

    def conditional_forward_step(self, x, t1, t2):
        """Must be implemented in subclass."""
        raise NotImplementedError(
            "This method should be overridden by subclass."
        )

    def sample_prior_v(self, x):
        """Must be implemented in subclass."""
        raise NotImplementedError(
            "This method should be overridden by subclass."
        )

    def velocity_step(self, x, t_id, backward=False):
        """Must be implemented in subclass."""
        raise NotImplementedError(
            "This method should be overridden by subclass."
        )

    def is_augmented(self):
        return False

    def process_params(self, model_type: str, model_params: dict) -> None:
        """
        Processes model parameters based on the model type and whether it is augmented.

        Parameters:
        - model_type (str): The type of the model.
        - model_params (dict): Dictionary containing model parameters.

        Returns:
        - None
        """
        is_augmented = self.is_augmented()
        if model_type == Case.u_net:
            if is_augmented and not "channel_out" in model_params:
                model_params["channels_out"] = model_params["image_channels"]
                model_params["image_channels"] = (
                    2 * model_params["image_channels"]
                )
        elif model_type == Case.vector_field:
            if "dim" in model_params:
                if is_augmented:
                    model_params["dim_in"] = 2 * model_params["dim"]
                else:
                    model_params["dim_in"] = model_params["dim"]
                model_params["dim_out"] = model_params["dim"]
                model_params.pop("dim")
        elif model_type == Case.ncsnpp:
            if is_augmented:
                model_params["v_input"] = True
        else:
            raise RuntimeError("Model type not implemented")

    @torch.no_grad()
    def forward_pass(
        self,
        x: torch.Tensor,
        return_trajectories: bool = False,
        use_training_velocity: bool = False,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Parameters:
        - x (torch.Tensor): The input tensor to the model.
        - return_trajectories (bool): Whether to return the entire trajectory or just the final state.
        - use_training_velocity (bool): Whether to use training-specific velocity.

        Returns:
        - torch.Tensor: The output tensor after the forward pass.
        """
        times, dts, nb_ts = (
            self.times_eval,
            self.dt_eval,
            self.nb_time_steps_eval,
        )
        x = self._augment_input(x)
        x_traj = self._initialize_trajectories(x, return_trajectories)
        save_idx_times = (
            self._get_traj_idx(forward=True) if return_trajectories else []
        )

        for i in range(nb_ts):
            t1, t2, dt = times[i], times[i + 1], dts[i]
            if not use_training_velocity:
                x = self.conditional_forward_step(x, t1, t2)
            else:
                x = self._step_velocity(x, t1, dt, i, backward=False)

            if return_trajectories and i + 1 in save_idx_times:
                append_trajectories(x_traj, x, self.is_augmented())

        return self._finalize_return(x, x_traj, return_trajectories)

    def particles_dim(self) -> Tuple[int, ...]:
        """
        Returns the dimensions of particles based on data type.
        """
        if self.data_type in toy_data_type:
            return (self.model_params["dim_out"],)
        else:
            return (
                self.model_params["image_channels"],
                self.model_params["dim"],
                self.model_params["dim"],
            )

    @torch.no_grad()
    def sample(
        self,
        nb_samples: Optional[int] = None,
        return_trajectories: bool = False,
        return_velocities: bool = False,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Samples from the model.

        Parameters:
            nb_samples (Optional[int]): Number of samples to generate. Either
              `nb_samples` or `x_init` should be provided.
            return_trajectories (bool): Whether to return trajectories.
            return_velocities (bool): Whether to return velocities.
            x_init (Optional[torch.Tensor]): Initial input for sampling.
              Either `nb_samples` or `x_init` should be provided.

        Returns:
            torch.Tensor: Sampled outputs.
        """
        if x_init is None and nb_samples is None:
            raise RuntimeError(
                "For sampling, either nb_samples or x_init should be provided."
            )

        if x_init is None:
            shape = (nb_samples,) + self.particles_dim()
            device = next(self.neural_network.parameters()).device
            x = torch.randn(shape, device=device)
        else:
            x = x_init

        return self.backward(
            x,
            return_trajectories=return_trajectories,
            return_velocities=return_velocities,
        )

    @torch.no_grad()
    def get_velocities(self, x: torch.Tensor) -> torch.Tensor:
        return self.backward(x, return_velocities=True)

    @torch.no_grad()
    def get_neural_net(self, x: torch.Tensor) -> torch.Tensor:
        return self.backward(x, return_neural_net=True)

    def backward_discrete_time(self) -> bool:
        """
        Indicates whether backward computation is made with discrete time.

        Returns:
            bool: True if backward computation is made with discrete time,
            else False.
        """
        return False

    def backward(
        self,
        x: torch.Tensor,
        return_trajectories: bool = False,
        return_velocities: bool = False,
        return_neural_net: bool = False,
    ) -> torch.Tensor:
        """
        Performs backward computation for the model.

        Parameters:
            x (torch.Tensor): Input data.
            return_trajectories (bool): Whether to return trajectories.
            return_velocities (bool): Whether to return velocities.
            return_neural_net (bool): Whether to return neural network outputs.

        Returns:
            torch.Tensor: Computed outputs. Varies based on flags.
        """
        self._validate_parameters(return_velocities, return_neural_net)

        if return_velocities or return_neural_net:
            return self._compute_velocities(x, return_neural_net)

        shape, dim = x.shape[0], x.dim()

        x = self._augment_input(x)

        x_traj = self._initialize_trajectories(x, return_trajectories)
        save_idx_times = (
            self._get_traj_idx()[:-1] if return_trajectories else []
        )

        for i in reversed(range(self.nb_time_steps_eval)):
            t, dt = self._get_time_info(
                i + 1, shape, dim, is_eval=True, backward_dt=True
            )
            x = self._step_velocity(x, t, dt, i, backward=True)
            if return_trajectories and i in save_idx_times:
                append_trajectories(x_traj, x, self.is_augmented())

        return self._finalize_return(x, x_traj, return_trajectories)

    def _validate_parameters(
        self, return_velocities: bool, return_neural_net: bool
    ) -> None:
        if return_velocities and return_neural_net:
            raise RuntimeError(
                "Both return_velocities and return_neural_net cannot be True at the same time"
            )

    def _augment_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_augmented():
            v = self.sample_prior_v(x)
            return x, v

        return x

    def _initialize_trajectories(
        self, x: torch.Tensor, return_trajectories: bool
    ):
        if return_trajectories:
            x_traj = [] if not self.is_augmented() else ([], [])
            append_trajectories(x_traj, x, self.is_augmented())
            return x_traj
        return None

    def _finalize_return(self, x, x_traj, return_trajectories: bool):
        if return_trajectories:
            return trajectories_to_array(x_traj, self.is_augmented())
        return x

    def _get_time_info(
        self,
        i: Union[int, torch.Tensor],
        shape: int = None,
        dim: int = None,
        is_eval: bool = True,
        backward_dt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get time information based on the current index and other parameters.

        Parameters:
            i (int): Current index.
            shape (int): The shape of the output tensor for time. Default to None.
            dim (int): The dummy dimension to be repeated. Default to None.
            is_eval (bool, optional): Whether to use evaluation times. Default
            to True.
            backward_dt (bool, optional): Whether to go backward in time and
            consider previous dt. Default to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The time and dt information.
        """
        # Determine the time and dt based on the flags
        dt_source = self.dt_eval if is_eval else self.dt_train
        time_source = self.times_eval if is_eval else self.times_train

        dt_id = i if not backward_dt else i - 1
        dt = dt_source[dt_id]
        t = time_source[i]
        if shape is not None and dim is not None:
            # Reshape the time output.
            t = t.repeat((shape,) + (dim - 1) * (1,))

        return t, dt

    def _step_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        i: int,
        backward: bool,
    ) -> torch.Tensor:
        if not self.backward_discrete_time():
            return self.velocity_step(x, t, dt, backward=backward)
        else:
            return self.velocity_step(x, i, backward=backward)

    def eval_nn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.neural_network(x, t)

    def _compute_velocities(
        self, x: torch.Tensor, return_neural_net: bool = False
    ) -> List[np.ndarray]:
        """
        Computes and returns velocities for the input tensor `x` at multiple
        time steps. The specific time steps are determined by the function
        `_get_traj_idx()`.

        If `return_neural_net` is True, the function also returns the neural
        network outputs associated with each velocity computation.

        Parameters:
            x (torch.Tensor): Input data tensor.
            return_neural_net (bool, optional): Flag indicating whether to
              return neural network outputs. Defaults to False.

        Returns:
            List[torch.Tensor]: A list of torch tensors containing the computed
            velocities.
        """
        velocities = []
        shape, dim = x.shape[0], x.dim()
        x = self._augment_input(x)

        save_idx_times = self._get_traj_idx()
        for t_id in save_idx_times:
            t = self.times_eval[t_id].repeat((shape,) + (dim - 1) * (1,))
            v = self._compute_velocity(
                x, t, return_neural_net=return_neural_net
            )

            velocities.append(v.cpu())
        velocities.reverse()
        return trajectories_to_array(velocities, False)

    def _compute_velocity(
        self, x: torch.Tensor, t: torch.Tensor, return_neural_net: bool = False
    ) -> torch.Tensor:
        """
        Computes the velocity or neural network outputs for the given input
        x and time t.

        Parameters:
            x (torch.Tensor): Input data.
            t (torch.Tensor): Time values.
            return_neural_net (bool): Whether to return neural network outputs
            instead of the velocity. Default to False.

        Returns:
            torch.Tensor: Computed velocity.
        """
        if not self.is_augmented():
            if not return_neural_net:
                velocity = self.velocity_eval(x, t)
            else:
                velocity = self.eval_nn(x, t)  # self.neural_network(x, t)
        else:
            x, v = x
            if not return_neural_net:
                velocity = self.velocity_eval(x, v, t)  # , v
            else:
                x_v = torch.cat((x, v), dim=-1)
                velocity = self.eval_nn(x_v, t)
        return velocity

    def get_traj_times(self, forward: bool = False) -> np.ndarray:
        """
        Get trajectory times for forward or backward computations.

        Parameters:
            forward (bool): Whether to get times for forward computations.
            Defaults to False.

        Returns:
            np.ndarray: Array trajectory times.
        """
        times_to_use = self.times_eval[:-1] if forward else self.times_eval[1:]

        return reduce_length_array(times_to_use.cpu().numpy())

    def _get_traj_idx(self, forward: bool = False) -> List[int]:
        """
        Get trajectory indices for forward or backward computations.

        Parameters:
            forward (bool): Whether to get indices for forward computations.
            Defaults to False.

        Returns:
            List[int]: List of trajectory indices.
        """
        if self.return_all_trajs:
            return list(range(self.nb_time_steps_eval))

        start_idx = 0 if forward else 1
        end_idx = self.times_eval.shape[0] - (1 if forward else 0)
        traj_idx = reduce_length_array(np.arange(start_idx, end_idx))
        return (traj_idx - 1) if forward else traj_idx
