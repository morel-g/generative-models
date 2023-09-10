import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import Dataset, get_dataset
from torchvision import transforms
from ..utils import id_to_device, get_logger
from torch.utils.data import Subset
from torch.utils.data import default_collate


class DataModule(pl.LightningDataModule):
    def __init__(self, data, log_dir=None):
        """Initialize data module.

        Args:
            data: A data object containing all the parameters of the simulation.
            logger: A logger if available to store the outputs. Defaults to None.
        """
        super().__init__()
        self.data = data
        self.device = (
            id_to_device(data.accelerator, data.device)
            if isinstance(data.device, list)
            else None
        )
        self.batch_size = data.training_params["batch_size"]
        self.batch_size_eval = data.training_params.get(
            "batch_size_eval", self.batch_size
        )
        self.pin_memory = False if (self.device == "cpu") else True
        self.map = None
        self.train_img_data, self.val_img_data = None, None
        n_samples = data.n_samples if hasattr(data, "n_samples") else None
        normalized_img = (
            data.normalized_img if hasattr(data, "normalized_img") else False
        )
        # Get the mean and std if needed
        (
            self.train_data,
            self.val_data,
        ) = get_dataset(
            self.data.data_type,
            log_dir,
            normalized_img=normalized_img,
            n_samples=n_samples,
        )
        self.custom_train_data = None
        self.custom_val_data = None
        self.use_custom_data = False

        if self.train_data.transform is not None:
            normalize = next(
                (
                    t
                    for t in self.train_data.transform.transforms
                    if isinstance(t, transforms.Normalize)
                ),
                None,
            )
        else:
            normalize = None
        self.mean_std = (
            (normalize.mean, normalize.std) if normalize is not None else None
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def collate_fn(self, dataset, batch):
        if hasattr(dataset, "update"):
            dataset.update()
        return default_collate(batch)

    def train_dataloader(self):
        """Return the dataloader of the encoded training set.

        Returns:
            Dataloader of the encoded training set.
        """
        # self.train_data.data = self.train_data.data[:50]
        # self.train_data.x.data = self.train_data.x.data[:50]
        data = (
            self.train_data
            if not self.use_custom_data
            else self.custom_train_data
        )
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: self.collate_fn(data, batch),
        )

    def val_dataloader(self):
        """Return the dataloader of the encoded validation set.

        Returns:
            Dataloader of the encoded validation set.
        """
        # self.val_data.data = self.val_data.data[:50]
        # self.val_data.x.data = self.val_data.x.data[:50]
        data = (
            self.val_data if not self.use_custom_data else self.custom_val_data
        )
        return DataLoader(
            data,
            batch_size=self.batch_size_eval,
            num_workers=2,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: self.collate_fn(data, batch),
        )

    def test_dataloader(self):
        return self.val_dataloader()
