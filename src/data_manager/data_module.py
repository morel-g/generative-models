import pytorch_lightning as pl
from src.utils import id_to_device
from omegaconf import DictConfig
from torch.utils.data import default_collate, Dataset as TorchDataset, DataLoader
from src.case import Case
from src.data_manager.data_type import (
    toy_continuous_data_type,
    toy_discrete_data_type,
    img_data_type,
    audio_data_type,
    text_data_type,
    rl_data_type,
    custom_data_type,
    manifold_data_type,
)
from src.data_manager.toy_data_utils import ToyDataUtils
from src.data_manager.img_data_utils import ImgDataUtils
from src.data_manager.text_data_utils import TextDataUtils
from src.data_manager.audio_data_utils import AudioDataUtils
from src.data_manager.rl_data_utils import RLDataUtils
from src.data_manager.custom_data_utils import CustomDataUtils
from src.data_manager.manifold_data_utils import ManifoldDataUtils


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, log_dir: str = None):
        """
        Initialize the data module.

        Parameters:
        - config (DictConfig): A DictConfig object containing all the parameters of the
        simulation.
        - log_dir (str, optional): Directory for logs. Defaults to None.
        """
        super().__init__()

        self.config = config
        self.device = (
            id_to_device(config.accelerator, config.device)
            if isinstance(config.device, list)
            else None
        )
        self.batch_size = config.training_params["batch_size"]
        self.batch_size_eval = config.training_params.get(
            "batch_size_eval", self.batch_size
        )
        self.pin_memory = self.device != "cpu"
        self.train_img_data, self.val_img_data = None, None
        n_samples = getattr(config, "n_samples", None)

        self.train_data, self.val_data = self.get_dataset(
            log_dir=log_dir,
            n_samples=n_samples,
        )
        self.custom_train_data = None
        self.custom_val_data = None
        self.use_custom_data = False

    def get_dataset(
        self,
        log_dir: str,
        n_samples: int = None,
    ) -> TorchDataset:
        """
        Fetches the dataset based on the provided data type and other parameters.

        Parameters:
        - log_dir (str): Directory for logs.
        - n_samples (int, optional): Number of samples, required for 2D datasets.
        Returns:
        - TorchDataset: The prepared dataset.
        """
        data_type = self.config.data_type

        kwargs = {}

        if (
            hasattr(self.config, "custom_data")
            and self.config.custom_data["use_custom_data"]
        ):
            kwargs["data_dir"] = self.config.custom_data["data_dir"]
            kwargs["prefix_1"] = self.config.custom_data["prefix_1"]
            kwargs["prefix_2"] = self.config.custom_data.get("prefix_2", None)
            kwargs["random_y_idx"] = self.config.custom_data.get("random_y_idx", False)
            return CustomDataUtils.prepare_custom_dataset(**kwargs)
        elif data_type in toy_continuous_data_type + toy_discrete_data_type:
            if self.config.data_type in toy_discrete_data_type:
                kwargs["nb_tokens"] = self.config.model_params["nb_tokens"]
            return ToyDataUtils.prepare_toy_dataset(
                data_type, n_samples, log_dir, **kwargs
            )
        elif data_type in audio_data_type:
            return AudioDataUtils.prepare_audio_dataset(data_type)
        elif data_type in img_data_type:
            return ImgDataUtils.prepare_img_dataset(data_type)
        elif data_type in text_data_type:
            kwargs["seq_length"] = self.config.scheme_params["seq_length"]
            kwargs["tokenizer_name"] = self.config.scheme_params.get(
                "tokenizer_name", Case.gpt2
            )
            return TextDataUtils.prepare_text_dataset(data_type, **kwargs)
        elif data_type in rl_data_type:
            if "horizon" in self.config.model_params:
                kwargs["horizon"] = self.config.model_params["horizon"]
            datasets = RLDataUtils.prepare_rl_dataset(data_type, **kwargs)
            if not "horizon" in self.config.model_params:
                self.config.model_params["horizon"] = RLDataUtils.horizon
            return datasets
        elif data_type in manifold_data_type:
            return ManifoldDataUtils.prepare_manifold_dataset(data_type)
        else:
            raise RuntimeError(f"Uknown data_type {data_type}")

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
        data = self.train_data if not self.use_custom_data else self.custom_train_data
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: self.collate_fn(data, batch),
            drop_last=True,
        )

    def val_dataloader(self):
        """Return the dataloader of the encoded validation set.

        Returns:
            Dataloader of the encoded validation set.
        """
        # self.val_data.data = self.val_data.data[:50]
        # self.val_data.x.data = self.val_data.x.data[:50]
        data = self.val_data if not self.use_custom_data else self.custom_val_data
        return DataLoader(
            data,
            batch_size=self.batch_size_eval,
            num_workers=2,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: self.collate_fn(data, batch),
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the dataloader of the encoded validation set.
        Note: This method currently uses the validation dataloader.

        Returns:
        - DataLoader: Dataloader of the encoded validation set.
        """
        return self.val_dataloader()

    def get_x_val(self):
        return self.val_data.x
