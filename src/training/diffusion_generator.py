import os
import torch
import pytorch_lightning as pl
from torch.optim.swa_utils import AveragedModel
from typing import Union, Tuple, Any, List, Dict, Optional

# from .probability_distribution import ProbabilityDistribution
from src.params import Params
from src.case import Case
from src.data_manager.data_type import (
    toy_data_type,
    img_data_type,
    audio_data_type,
    text_data_type,
    rl_data_type,
)
from src.models.score_model import ScoreModel
from src.models.score_model_critical_damped import ScoreModelCriticalDamped
from src.models.stochastic_interpolant import StochasticInterpolant
from src.models.d3pm import D3PM
from src.eval.plots_2d import sample_2d
from src.eval.plots_img import sample_img
from src.eval.plots_text import sample_text
from src.eval.plots_rl import sample_rl
from src.training.opt_utils import create_optimizer, create_scheduler
from src.training.ema_handler import EMAHandler
from src.data_manager.rl_data_utils import RLDataUtils


class DiffusionGenerator(pl.LightningModule):
    """
    This class handle the diffusion process
    """

    def __init__(
        self,
        params: Params,
    ) -> None:
        """
        Initialize the DiffusionGenerator model.

        Parameters:
            params: Configuration and scheme parameters for the diffusion generator.
        """
        super(DiffusionGenerator, self).__init__()
        self.save_hyperparameters()
        self.params = params
        self.ema_dict = params.training_params.get("ema_dict", {"use_ema": False})
        self.ema = self.ema_dict["use_ema"]
        if params.model_type == Case.score_model:
            self.base_model = ScoreModel(
                params.data_type,
                params.model_params,
                params.scheme_params["nb_time_steps_eval"],
                params.scheme_params["nb_time_steps_train"],
                T_final=params.scheme_params["T_final"],
                beta_case=params.scheme_params.get("beta_case", Case.vanilla),
                adapt_dt=params.scheme_params["adapt_dt"],
                pde_coefs=params.scheme_params.get(
                    "pde_coefs",
                    {"gamma": 1.0},
                ),
                decay_case=params.scheme_params.get("decay_case", Case.vanilla_sigma),
                img_model_case=params.scheme_params.get("img_model_case", Case.u_net),
                conditioning_case=params.scheme_params.get("conditioning_case", None),
            )
        elif params.model_type == Case.score_model_critical_damped:
            self.base_model = ScoreModelCriticalDamped(
                params.data_type,
                params.model_params,
                params.scheme_params["nb_time_steps_eval"],
                params.scheme_params["nb_time_steps_train"],
                T_final=params.scheme_params["T_final"],
                adapt_dt=params.scheme_params["adapt_dt"],
                decay_case=params.scheme_params.get("decay_case", Case.vanilla_sigma),
                img_model_case=params.scheme_params.get("img_model_case", Case.u_net),
                init_var_v=params.scheme_params.get("init_var_v", 0.04),
                zeros_S0_vv=params.scheme_params.get("zeros_S0_vv", False),
            )
        elif params.model_type == Case.stochastic_interpolant:
            self.base_model = StochasticInterpolant(
                params.data_type,
                params.model_params,
                params.scheme_params["nb_time_steps_eval"],
                params.scheme_params["nb_time_steps_train"],
                T_final=params.scheme_params["T_final"],
                beta_case=params.scheme_params["beta_case"],
                adapt_dt=params.scheme_params["adapt_dt"],
                decay_case=params.scheme_params.get("decay_case", Case.exp),
                interpolant=params.scheme_params["interpolant"],
                img_model_case=params.scheme_params.get("img_model_case", Case.u_net),
                noise_addition=params.scheme_params.get("noise_addition", None),
            )
        elif params.model_type == Case.d3pm:
            self.base_model = D3PM(
                params.data_type,
                params.model_params,
                seq_length=params.scheme_params["seq_length"],
                nb_time_steps_eval=params.scheme_params["nb_time_steps_eval"],
                nb_time_steps_train=params.scheme_params["nb_time_steps_train"],
                T_final=params.scheme_params["T_final"],
                transition_case=params.scheme_params.get(
                    "transition_case", Case.uniform
                ),
                store_transition_matrices=params.scheme_params.get(
                    "store_transition_matrices", False
                ),
            )
        else:
            raise ValueError("Model type not recognized")

        if self.ema:
            ema_config = {k: v for k, v in self.ema_dict.items() if k != "use_ema"}
            self.ema_handler = EMAHandler(**ema_config)
            self.ema_model = AveragedModel(
                self.base_model, multi_avg_fn=self.ema_handler.ema_multi_avg_fn
            )

    def get_model(self):
        if not self.ema:
            return self.base_model
        elif self.training:
            return self.base_model
        else:
            return self.ema_model.module

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
        if self.params.data_type in img_data_type and len(batch) == 2:
            x, _ = batch
        else:
            x = batch
        model = self.get_model()
        loss = model.loss(x)

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
        model = self.get_model()
        if self.params.data_type in img_data_type and len(batch) == 2:
            x, _ = batch
        else:
            x = batch
        loss = model.loss(x).cpu().item()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    @torch.no_grad()
    def sample(
        self,
        nb_samples: Optional[int] = None,
        return_trajectories: bool = False,
        return_velocities: bool = False,
        x_init: Optional[torch.Tensor] = None,
        x_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample data points using the model.

        Parameters:
            nb_samples (Optional[int]): Number of samples to generate.
            return_trajectories: Whether to return trajectories.
            return_velocities: Whether to return velocities.
            x_init (Optional[torch.Tensor]): Initial input for sampling.
              Either `nb_samples` or `x_init` should be provided.
            x_cond (Optional[torch.Tensor]): Conditional data, if any.

        Returns:
            x: The sampled data points.
        """
        model = self.get_model()
        x = model.sample(
            nb_samples,
            return_trajectories,
            return_velocities=return_velocities,
            x_init=x_init,
            x_cond=x_cond,
        )

        if self.params.data_type in img_data_type:
            if model.is_augmented():
                x, v = x
                x, v = (x + 1.0) / 2.0, (v + 1.0) / 2.0
                x = x, v
            else:
                x = (x + 1.0) / 2.0
        elif self.params.data_type in rl_data_type:
            x = RLDataUtils.denormalize(x)

        return x

    def is_augmented(self) -> bool:
        """
        Check if the model is augmented.

        Returns:
            bool: True if augmented, False otherwise.
        """
        model = self.get_model()
        return model.is_augmented()

    def set_adapt_dt(self, adapt_dt: bool) -> None:
        model = self.get_model()
        model.adapt_dt = adapt_dt

    def set_backward_scheme(self, scheme: str) -> None:
        model = self.get_model()
        model.backward_scheme = scheme

    def set_trajectory_length(self, trajectory_length: int) -> None:
        model = self.get_model()
        model.trajectory_length = trajectory_length

    def set_nb_time_steps(self, nb_time_steps: int, eval: bool = False):
        model = self.get_model()
        model.set_nb_time_steps(nb_time_steps, eval=eval)

    def get_velocities(self, x: torch.Tensor) -> torch.Tensor:
        model = self.get_model()
        return model.get_velocities(x.to(self.device))

    def get_neural_net(self, x: torch.Tensor) -> torch.Tensor:
        model = self.get_model()
        return model.get_neural_net(x.to(self.device))

    def get_nb_time_steps_eval(self) -> int:
        model = self.get_model()
        return model.get_nb_time_steps_eval()

    def get_backward_schemes(self) -> List[str]:
        model = self.get_model()
        return model.get_backward_schemes()

    def get_backward_scheme(self) -> str:
        model = self.get_model()
        return model.backward_scheme

    def get_traj_times(self) -> torch.Tensor:
        model = self.get_model()
        return model.get_traj_times()

    def get_samples_prior(self, nb_samples):
        model = self.get_model()
        return model.get_samples_prior(nb_samples)

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
        model = self.get_model()
        return model.forward_pass(
            x.to(self.device), return_trajectories, use_training_velocity
        )

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Dict]]]:
        """
        Configure the optimizer and the learning rate scheduler for the
        PyTorch Lightning training loop.

        Returns:
        - Either a single optimizer or a tuple containing a list of
        optimizers and a list of scheduler dictionaries.
        """
        optimizer = create_optimizer(self.base_model, self.params.training_params)
        scheduler = create_scheduler(
            optimizer,
            self.params.training_params,
            data_module=self.trainer.datamodule,
        )

        if scheduler is not None:
            return [optimizer], [scheduler]

        return optimizer

    def on_train_start(self):
        if self.ema:
            self.ema_model.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema:
            self.ema_model.update_parameters(self.base_model)

    def on_train_end(self):
        if self.ema:
            torch.optim.swa_utils.update_bn(
                self.trainer.datamodule.train_dataloader(), self.ema_model
            )

    def prepare_for_inference(self, train_dataloader, device: str) -> None:
        """
        Prepares the model for inference by updating BatchNorm statistics.

        Parameters:
        - train_dataloader: The data loader containing the training data used to update statistics.
        - device (str): The device to which the model should be moved (e.g., 'cpu' or 'cuda').
        """
        self.eval()
        self.to(device)
        if self.ema:
            torch.optim.swa_utils.update_bn(train_dataloader, self.ema_model)
            self.ema_model.to(device)

    def save_validation_samples(self) -> None:
        """
        Saves validation samples generated by the model.
        """
        sample_name = f"Samples_epoch_{self.trainer.current_epoch}"
        sample_path = os.path.join(
            self.trainer.checkpoint_callback.dirpath, "training_samples"
        )

        # Check the data type and call the appropriate sampling function
        if self.params.data_type in toy_data_type:
            sample_2d(self, sample_path, sample_name)
        elif self.params.data_type in img_data_type:
            if self.params.data_type in audio_data_type:
                nb_rows = 2
                nb_cols = 3
            else:
                nb_rows = 5
                nb_cols = 5
            sample_img(
                self,
                sample_path,
                sample_name,
                nb_rows=nb_rows,
                nb_cols=nb_cols,
                save_gifs=False,
            )
        elif self.params.data_type in text_data_type:
            sample_text(self, sample_path, sample_name, nb_samples=5)
        elif self.params.data_type in rl_data_type:
            sample_rl(self, sample_path, sample_name, nb_samples=6)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Handles actions to take on saving a checkpoint.

        Parameters:
            checkpoint (Dict[str, Any]): The checkpoint data to save.
        """
        super().on_save_checkpoint(checkpoint)
        if self.ema:
            checkpoint["ema_step_counter"] = self.ema_handler.step
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        self.save_validation_samples()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Restores model state from a checkpoint.

        Parameters:
        - checkpoint (Dict[str, Any]): A dictionary containing the checkpoint data.

        Raises:
        - RuntimeError: If EMA weights are enabled but not found in the checkpoint.
        """
        super().on_load_checkpoint(checkpoint)
        if (
            self.ema
            and "ema_state_dict" in checkpoint
            and "ema_step_counter" in checkpoint
        ):
            self.ema_handler.step = checkpoint["ema_step_counter"]
            ema_state_dict = checkpoint.get("ema_state_dict")
            self.ema_model.load_state_dict(ema_state_dict)
            print(f"EMA successfully loaded at step {self.ema_handler.step}")
        elif self.ema:
            raise RuntimeError(
                "No ema_state_dict found in checkpoint. EMA weights can't be restored."
            )
