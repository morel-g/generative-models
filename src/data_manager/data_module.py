import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import get_dataset
from torchvision import transforms
from ..utils import id_to_device
from torch.utils.data import default_collate


class DataModule(pl.LightningDataModule):
    def __init__(self, data: object, log_dir: str = None):
        """
        Initialize the data module.

        Parameters:
        - data (object): A data object containing all the parameters of the
        simulation.
        - log_dir (str, optional): Directory for logs. Defaults to None.
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
        self.pin_memory = self.device != "cpu"
        self.train_img_data, self.val_img_data = None, None
        n_samples = getattr(data, "n_samples", None)

        self.train_data, self.val_data = get_dataset(
            data_type=self.data.data_type,
            log_dir=log_dir,
            n_samples=n_samples,
        )
        self.custom_train_data = None
        self.custom_val_data = None
        self.use_custom_data = False

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def collate_fn(self, dataset: object, batch: list) -> list:
        """
        Custom collation function for batching data.

        Parameters:
        - dataset (object): Dataset object to collate.
        - batch (list): List of data to collate.

        Returns:
        - list: Collated batch.
        """
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

    def test_dataloader(self) -> DataLoader:
        """
        Return the dataloader of the encoded validation set.
        Note: This method currently uses the validation dataloader.

        Returns:
        - DataLoader: Dataloader of the encoded validation set.
        """
        return self.val_dataloader()
