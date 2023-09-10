from src.models.ebm.ebm_model import EBMModel
from src.case import Case
from src.data_manager.data_type import toy_data_type
from src.eval.plots_2d import sample_2d
from src.eval.plots import sample_img
from src.density_generation import DensityGeneration

import pytorch_lightning as pl
import torch
import random
from torch.utils.data import Dataset
from diffusers.optimization import get_cosine_schedule_with_warmup


def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[: axis + 1]).argsort(
        axis
    )  # get permutation indices
    for _ in range(tensor.ndim - axis - 1):
        row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(
        *[1 for _ in range(axis + 1)], *(tensor.shape[axis + 1 :])
    )  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)


class CustomDataset(Dataset):
    def __init__(self, x):
        self.x = x
        self.set_random_time_step()

    def set_random_time_step(self):
        self.time_step = random.randint(0, self.x.shape[0] - 2)

    def update(self):
        self.set_random_time_step()

    def __getitem__(self, index):
        return self.x[:, index, ...]

    def __len__(self):
        return self.x.shape[1]


class CustomDataset2(Dataset):
    def __init__(self, x):
        self.x = x
        self.set_random_time_step()

    def set_random_time_step(self):
        self.time_step = random.randint(0, self.x.shape[0] - 2)

    def update(self):
        self.set_random_time_step()

    def __getitem__(self, index):
        # index = index % self.x.shape[1]
        return (
            self.x[self.time_step][index],
            self.x[self.time_step + 1][index],
            self.time_step,
        )

    def __len__(self):
        return self.x.shape[1]


class EBMGenerator(pl.LightningModule):
    def __init__(self, data):
        super().__init__()
        self.save_hyperparameters()
        self.data = data
        self.alpha = data.training_params.get("alpha", 0.1)

        self.beta1 = data.training_params.get("beta1", 0.0)
        self.model = EBMModel(
            data.data_type,
            data.model_params,
            data.scheme_params["nb_time_steps_eval"],
            data.scheme_params["nb_time_steps_train"],
            T_final=data.scheme_params["T_final"],
            beta_case=data.scheme_params["beta_case"],
            adapt_dt=data.scheme_params["adapt_dt"],
            pde_coefs=data.scheme_params.get(
                "pde_coefs",
                {"gamma": 1.0},
            ),
            decay_case=data.scheme_params.get("decay_case", Case.exp),
            img_model_case=data.scheme_params.get(
                "img_model_case", Case.u_net
            ),
            batch_size=data.training_params["batch_size"],
            inter_it_dt=data.scheme_params.get("inter_it_dt", 60),
        )
        print(
            "Using density generation with fokker planck samples NOT BOLTZMAN"
        )
        self.density_gen = DensityGeneration(
            Case.score_model,  # self.testing_params["generative_model"]
        )
        self.use_all_times_at_once = (
            True  # CustomDataset if True else CustomDataset2
        )

    def remove_titles(self):
        return True

    def on_fit_start(self):
        x_val = self.trainer.datamodule.val_data.x.to(self.device)
        if x_val.dim() != 2:
            raise RuntimeError(
                "Only implemented in 2d (dataset do not generalize to image)"
            )
        dt = self.model.dt_train
        x = self.density_gen.generate(x_val, dt)
        dataset = (
            CustomDataset2 if not self.use_all_times_at_once else CustomDataset
        )
        self.trainer.datamodule.custom_val_data = dataset(x)
        x_train = self.trainer.datamodule.train_data.x.to(self.device)
        x = self.density_gen.generate(x_train, dt)
        self.trainer.datamodule.custom_train_data = dataset(x)
        self.trainer.datamodule.use_custom_data = True

    def on_fit_end(self):
        self.trainer.datamodule.use_custom_data = False
        self.trainer.datamodule.custom_train_data = None
        self.trainer.datamodule.custom_val_data = None

    def on_train_epoch_start(self):
        # nb_it_velocity = self.custom_path_dict.get(
        #     "nb_epochs_train_velocity", 5
        # )
        # nb_it_path = self.custom_path_dict.get("nb_epochs_train_path", 5)
        # counter = (self.current_epoch) % (nb_it_velocity + nb_it_path)
        # if self.train_path:
        #     if (counter < nb_it_velocity) != self.train_velocity:
        #         print(f"\n Train velocity: {counter < nb_it_velocity}")
        #     self.train_velocity = counter < nb_it_velocity
        # else:
        #     self.train_velocity = True

        if self.current_epoch % 10 == 0:
            print("Updating Boltzman samples")
            x_train = self.trainer.datamodule.train_data.x.to(self.device)
            if x_train.dim() != 2:
                raise RuntimeError(
                    "Only implemented in 2d (dataset do not generalize to image)"
                )
            dt = self.model.dt_train

            x = self.density_gen.generate(x_train, dt)
            dataset = (
                CustomDataset2
                if not self.use_all_times_at_once
                else CustomDataset
            )
            self.trainer.datamodule.custom_train_data = dataset(x)
        elif self.current_epoch % 2 == 0:
            self.trainer.datamodule.custom_train_data.x = shufflerow(
                self.trainer.datamodule.custom_train_data.x, 1
            )

    def training_step(self, batch, batch_idx):
        # x = batch[0] if len(batch) == 2 else batch
        # t_id = torch.zeros(x.shape[0], dtype=torch.int, device=x.device)
        if not self.use_all_times_at_once:
            x, y, t_id = batch
            reg_loss, cdiv_loss, y_real, y_fake = self.model.loss(x, y, t_id)
        else:
            x = batch
            reg_loss, cdiv_loss, y_real, y_fake = self.model.loss_2(x)
        loss = self.alpha * reg_loss + cdiv_loss

        # Logging
        self.log("loss", loss, prog_bar=True)
        self.log("loss_regularization", self.alpha * reg_loss)
        self.log("loss_contrastive_divergence", cdiv_loss)
        self.log("metrics_avg_real", y_real.mean())
        self.log("metrics_avg_fake", y_fake.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.use_all_times_at_once:
            x, y, t_id = batch
            reg_loss, cdiv_loss, y_real, y_fake = self.model.loss(x, y, t_id)
        else:
            x = batch
            reg_loss, cdiv_loss, y_real, y_fake = self.model.loss_2(x)
        self.log("val_loss", cdiv_loss + self.alpha * reg_loss, prog_bar=True)
        self.log("val_contrastive_divergence", cdiv_loss)
        self.log("val_fake_out", y_fake.mean())
        self.log("val_real_out", y_real.mean())

    @torch.no_grad()
    def sample(
        self,
        nb_samples,
        return_trajectories=False,
        save_score_pde_error=False,
        return_velocities=False,
    ):
        x = self.model.sample(
            nb_samples,
            return_trajectories,
            save_score_pde_error=save_score_pde_error,
            return_velocities=return_velocities,
        )

        return x

    def set_adapt_dt(self, adapt_dt):
        self.model.adapt_dt = adapt_dt

    def set_backward_scheme(self, scheme):
        self.model.backward_scheme = scheme

    def get_nb_time_steps_eval(self):
        return self.model.get_nb_time_steps_eval()

    def get_backward_schemes(self):
        return self.model.get_backward_schemes()

    def get_backward_scheme(self):
        return self.model.backward_scheme

    def get_traj_times(self, forward=False):
        return self.model.get_traj_times(forward)

    def forward_pass(
        self, x, return_trajectories=False, use_training_velocity=False
    ):
        return self.model.forward_pass(
            x, return_trajectories, use_training_velocity
        )

    def set_nb_time_steps(self, nb_time_steps, eval=False):
        self.model.set_nb_time_steps(nb_time_steps, eval=eval)

    def get_velocities(self, x):
        return self.model.get_velocities(x.to(self.device))

    def get_neural_net(self, x):
        return self.model.get_neural_net(x.to(self.device))

    def is_augmented(self):
        return self.model.is_augmented()

    def configure_optimizers(self):
        # Get the mean and std
        self.mean_std = self.trainer.datamodule.mean_std

        # Set optimzizer param
        training_params = self.data.training_params
        scheduler_dict = training_params.get("scheduler_dict", {})
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
            betas=(self.beta1, 0.999),
        )

        scheduler = scheduler_dict.get("scheduler", None)
        if scheduler == Case.cosine_with_warmup:
            batch_size = self.data.training_params["batch_size"]
            # Numer of steps in one epoch
            nb_steps_epoch = (
                len(self.trainer.datamodule.train_data) // batch_size
            )
            if len(self.trainer.datamodule.train_data) % batch_size != 0:
                nb_steps_epoch += 1

            sch = get_cosine_schedule_with_warmup(
                optimizer=opt,
                num_warmup_steps=scheduler_dict["num_warmup_epochs"]
                * nb_steps_epoch,
                num_training_steps=self.data.training_params["epochs"]
                * nb_steps_epoch,
            )
            scheduler = {
                "scheduler": sch,
                "interval": "step",  # or 'epoch'
                "frequency": 1,
            }
        elif scheduler is not None:
            raise RuntimeError("Unkown scheduler ", scheduler)

        return opt if scheduler is None else ([opt], [scheduler])

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        epoch = self.trainer.current_epoch
        name = "Samples_epoch_" + str(epoch)
        if self.data.data_type in toy_data_type:
            sample_2d(self, checkpoint_dir + "/training_samples", name)
        else:
            sample_img(
                self,
                checkpoint_dir + "/training_samples",
                name,
                nb_rows=5,
                nb_cols=5,
                save_gifs=False,
            )
