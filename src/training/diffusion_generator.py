import os
import torch
import pytorch_lightning as pl
from typing import Union, Tuple, Optional, Any, List, Dict

# from .probability_distribution import ProbabilityDistribution
from src.params import Params
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
from torch.optim.swa_utils import AveragedModel


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
        self.ema = params.training_params.get("ema", False)
        self.ema_rate = params.training_params.get("ema_rate", 0.999)
        self.ema_interval = params.training_params.get(
            "ema_interval", Case.epoch
        )

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
                decay_case=params.scheme_params.get(
                    "decay_case", Case.vanilla_sigma
                ),
                img_model_case=params.scheme_params.get(
                    "img_model_case", Case.u_net
                ),
            )
        elif params.model_type == Case.score_model_critical_damped:
            self.base_model = ScoreModelCriticalDamped(
                params.data_type,
                params.model_params,
                params.scheme_params["nb_time_steps_eval"],
                params.scheme_params["nb_time_steps_train"],
                T_final=params.scheme_params["T_final"],
                adapt_dt=params.scheme_params["adapt_dt"],
                decay_case=params.scheme_params.get(
                    "decay_case", Case.vanilla_sigma
                ),
                img_model_case=params.scheme_params.get(
                    "img_model_case", Case.u_net
                ),
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
                img_model_case=params.scheme_params.get(
                    "img_model_case", Case.u_net
                ),
                noise_addition=params.scheme_params.get(
                    "noise_addition", None
                ),
            )
        else:
            raise ValueError("Model type not recognized")

        if self.ema:
            self.ema_model = AveragedModel(
                self.base_model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                    self.ema_rate
                ),
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
        # if len(batch) == 2:
        #     x, _ = batch
        # else:
        #     x = batch
        model = self.get_model()
        x = batch
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
        # if len(batch) == 2:
        #     x, _ = batch
        # else:
        #     x = batch
        model = self.get_model()
        x = batch
        loss = model.loss(x).cpu().item()
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
        model = self.get_model()
        x = model.sample(
            nb_samples,
            return_trajectories,
            return_velocities=return_velocities,
        )

        if self.params.data_type in img_data_type:
            if model.is_augmented():
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
        model = self.get_model()
        return model.is_augmented()

    def set_adapt_dt(self, adapt_dt: bool) -> None:
        model = self.get_model()
        model.adapt_dt = adapt_dt

    def set_backward_scheme(self, scheme: str) -> None:
        model = self.get_model()
        model.backward_scheme = scheme

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
        optimizer = create_optimizer(
            self.base_model, self.params.training_params
        )
        scheduler = create_scheduler(
            optimizer,
            self.params.training_params,
            data_module=self.trainer.datamodule,
        )

        if scheduler is not None:
            return [optimizer], [scheduler]

        return optimizer

    # def on_validation_start(self):
    #     if self.ema:
    #         self.ema_model.to(self.device)

    # def on_validation_end(self):
    #     if self.ema:
    #         self.ema_model.cpu()

    def on_train_start(self):
        if self.ema:
            self.ema_model.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema and self.ema_interval == Case.step:
            self.ema_model.update_parameters(self.base_model)

    def on_train_epoch_end(self):
        # Update EMA parameters
        if self.ema and self.ema_interval == Case.epoch:
            self.ema_model.update_parameters(self.base_model)

    def on_train_end(self):
        # Update Batch Normalization statistics for the EMA model
        # using some DataLoader (here assumed to be part of a DataModule)
        torch.optim.swa_utils.update_bn(
            self.trainer.datamodule.train_dataloader(), self.ema_model
        )
        # self.update_ema_bn(self.trainer.datamodule.train_dataloader())

    def prepare_for_inference(self, train_dataloader, device):
        """
        This method updates BatchNorm statistics and prepares the model for inference.
        Call this method after loading the model and before performing inference tasks.
        """
        self.eval()
        self.to(device)
        if self.ema:
            # Update Batch Normalization statistics for the EMA model
            torch.optim.swa_utils.update_bn(train_dataloader, self.ema_model)
            self.ema_model.to(device)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Handles actions to take on saving a checkpoint.

        Parameters:
            checkpoint: The checkpoint data to save.
        """
        super().on_save_checkpoint(checkpoint)
        if self.ema:
            print("Save ema parameters")
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        epoch = self.trainer.current_epoch
        name = f"Samples_epoch_{epoch}"

        # Generate a full path for saving the sample
        sample_path = os.path.join(checkpoint_dir, "training_samples")

        # Check the data type and call the appropriate sampling function
        if self.params.data_type in toy_data_type:
            sample_2d(self, sample_path, name)
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
                name,
                nb_rows=nb_rows,
                nb_cols=nb_cols,
                save_gifs=False,
            )

    def on_load_checkpoint(self, checkpoint):
        # Call the parent's on_load_checkpoint
        super().on_load_checkpoint(checkpoint)
        if self.ema:
            print("Restore ema parameters")
            # Restore the state of the EMA model
            ema_state_dict = checkpoint.get("ema_state_dict")
            if ema_state_dict:
                self.ema_model.load_state_dict(ema_state_dict)
