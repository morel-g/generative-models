import os
import torch
import pytorch_lightning as pl
from typing import Union, Tuple, Optional, Any, List, Dict

# from .probability_distribution import ProbabilityDistribution
from src.data_manager.data import Data
from src.case import Case
from src.data_manager.data_type import (
    toy_data_type,
    img_data_type,
    audio_data_type,
)
from src.models.score_model import ScoreModel
from src.models.score_model_critical_damped import ScoreModelCriticalDamped
from src.models.stochastic_interpolant import StochasticInterpolant
from src.eval.plots_2d import sample_2d
from src.eval.plots import sample_img
from src.training.opt_utils import create_optimizer, create_scheduler


class DiffusionGenerator(pl.LightningModule):
    """
    This class handle the diffusion process
    """

    def __init__(
        self,
        data: Data,
    ) -> None:
        """
        Initialize the DiffusionGenerator model.

        Parameters:
            data: Configuration and scheme parameters for the diffusion generator.
        """
        super(DiffusionGenerator, self).__init__()
        self.save_hyperparameters()
        self.data = data

        if data.model_type == Case.score_model:
            self.model = ScoreModel(
                data.data_type,
                data.model_params,
                data.scheme_params["nb_time_steps_eval"],
                data.scheme_params["nb_time_steps_train"],
                T_final=data.scheme_params["T_final"],
                beta_case=data.scheme_params.get("beta_case", Case.vanilla),
                adapt_dt=data.scheme_params["adapt_dt"],
                pde_coefs=data.scheme_params.get(
                    "pde_coefs",
                    {"gamma": 1.0},
                ),
                decay_case=data.scheme_params.get(
                    "decay_case", Case.vanilla_sigma
                ),
                img_model_case=data.scheme_params.get(
                    "img_model_case", Case.u_net
                ),
            )
        elif data.model_type == Case.score_model_critical_damped:
            self.model = ScoreModelCriticalDamped(
                data.data_type,
                data.model_params,
                data.scheme_params["nb_time_steps_eval"],
                data.scheme_params["nb_time_steps_train"],
                T_final=data.scheme_params["T_final"],
                adapt_dt=data.scheme_params["adapt_dt"],
                decay_case=data.scheme_params.get(
                    "decay_case", Case.vanilla_sigma
                ),
                img_model_case=data.scheme_params.get(
                    "img_model_case", Case.u_net
                ),
                init_var_v=data.scheme_params.get("init_var_v", 0.04),
                zeros_S0_vv=data.scheme_params.get("zeros_S0_vv", False),
            )
        elif data.model_type == Case.stochastic_interpolant:
            self.model = StochasticInterpolant(
                data.data_type,
                data.model_params,
                data.scheme_params["nb_time_steps_eval"],
                data.scheme_params["nb_time_steps_train"],
                T_final=data.scheme_params["T_final"],
                beta_case=data.scheme_params["beta_case"],
                adapt_dt=data.scheme_params["adapt_dt"],
                decay_case=data.scheme_params.get("decay_case", Case.exp),
                interpolant=data.scheme_params["interpolant"],
                img_model_case=data.scheme_params.get(
                    "img_model_case", Case.u_net
                ),
                noise_addition=data.scheme_params.get("noise_addition", None),
            )
        else:
            raise ValueError("Model type not recognized")

    def training_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, Any]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Compute the loss for a training step.

        Parameters:
            batch: The input data batch.
            batch_idx: Index of the batch.

        Returns:
            loss: The computed loss.
        """
        if len(batch) == 2:
            x, _ = batch
        else:
            x = batch

        loss = self.model.loss(x)

        self.log("loss", loss, prog_bar=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, Any]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Compute and log the validation loss.

        Parameters:
            batch: The input data batch.
            batch_idx: Index of the batch.
        """
        if len(batch) == 2:
            x, _ = batch
        else:
            x = batch

        loss = self.model.loss(x).cpu().item()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    @torch.no_grad()
    def sample(
        self,
        nb_samples: int,
        return_trajectories: bool = False,
        return_velocities: bool = False,
    ) -> torch.Tensor:
        """
        Sample data points using the model.

        Parameters:
            nb_samples: Number of samples to generate.
            return_trajectories: Whether to return trajectories.
            return_velocities: Whether to return velocities.

        Returns:
            x: The sampled data points.
        """
        x = self.model.sample(
            nb_samples,
            return_trajectories,
            return_velocities=return_velocities,
        )

        if self.data.data_type in img_data_type:
            if self.model.is_augmented():
                x, v = x
                x, v = (x + 1.0) / 2.0, (v + 1.0) / 2.0
                x = x, v
            else:
                x = (x + 1.0) / 2.0

        return x

    def is_augmented(self) -> bool:
        """
        Check if the model is augmented.

        Returns:
            bool: True if augmented, False otherwise.
        """
        return self.model.is_augmented()

    def set_adapt_dt(self, adapt_dt: bool) -> None:
        self.model.adapt_dt = adapt_dt

    def set_backward_scheme(self, scheme: str) -> None:
        self.model.backward_scheme = scheme

    def set_nb_time_steps(self, nb_time_steps: int, eval: bool = False):
        self.model.set_nb_time_steps(nb_time_steps, eval=eval)

    def get_velocities(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_velocities(x.to(self.device))

    def get_neural_net(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_neural_net(x.to(self.device))

    def get_nb_time_steps_eval(self) -> int:
        return self.model.get_nb_time_steps_eval()

    def get_backward_schemes(self) -> List[str]:
        return self.model.get_backward_schemes()

    def get_backward_scheme(self) -> str:
        return self.model.backward_scheme

    def get_traj_times(self, forward: bool = False) -> torch.Tensor:
        return self.model.get_traj_times(forward)

    def forward_pass(
        self,
        x: torch.Tensor,
        return_trajectories: bool = False,
        use_training_velocity: bool = False,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Parameters:
        - x (torch.Tensor): The input tensor to the model.
        - return_trajectories (bool, optional): Flag indicating whether to return
          the entire trajectory of states. Defaults to False.
        - use_training_velocity (bool, optional): Flag indicating whether to use
          training-specific velocity parameters. Defaults to False.

        Returns:
        - torch.Tensor: The output tensor from the model.
        """
        return self.model.forward_pass(
            x, return_trajectories, use_training_velocity
        )

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Dict]]
    ]:
        """
        Configure the optimizer and the learning rate scheduler for the
        PyTorch Lightning training loop.

        Returns:
        - Either a single optimizer or a tuple containing a list of
        optimizers and a list of scheduler dictionaries.
        """
        optimizer = create_optimizer(self.model, self.data.training_params)
        scheduler = create_scheduler(
            optimizer,
            self.data.training_params,
            data_module=self.trainer.datamodule,
        )

        if scheduler is not None:
            return [optimizer], [scheduler]

        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Handles actions to take on saving a checkpoint.

        Parameters:
            checkpoint: The checkpoint data to save.
        """
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        epoch = self.trainer.current_epoch
        name = f"Samples_epoch_{epoch}"

        # Generate a full path for saving the sample
        sample_path = os.path.join(checkpoint_dir, "training_samples")

        # Check the data type and call the appropriate sampling function
        if self.data.data_type in toy_data_type:
            sample_2d(self, sample_path, name)
        elif self.data.data_type in img_data_type:
            if self.data.data_type in audio_data_type:
                nb_rows = 2
                nb_cols = 3
            else:
                nb_rows = 5
                nb_cols = 5
            sample_img(
                self,
                sample_path,
                name,
                nb_rows=nb_rows,
                nb_cols=nb_cols,
                save_gifs=False,
            )
