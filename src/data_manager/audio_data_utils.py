import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from datasets import load_dataset
from diffusers import Mel
from typing import Tuple
from src.case import Case


class AudioDiffusionDataset(Dataset):
    """
    A dataset class for audio diffusion datasets.
    """

    def __init__(self, huggingface_dataset: Dataset, transform=None):
        """
        Initializes the AudioDiffusionDataset with a dataset and an optional transform.

        Parameters:
            huggingface_dataset (Dataset): The dataset to use.
            transform (Optional[Callable]): The transform to apply on the data.
        """
        self.huggingface_dataset = huggingface_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.huggingface_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieves an item from the dataset at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, str]: A tuple containing the image tensor and the audio file path.
        """
        image = self.huggingface_dataset[idx]["image"]
        audio_path = self.huggingface_dataset[idx]["audio_file"]

        if self.transform:
            image = self.transform(image)

        return image  # , audio_path


class AudioDataUtils:
    """
    A utility class for audio data operations.
    """

    @staticmethod
    def prepare_audio_dataset(
        name: str,
    ) -> Tuple[AudioDiffusionDataset, AudioDiffusionDataset]:
        """
        Prepares the audio dataset for training and testing.

        Parameters:
            name (str): The name of the dataset.

        Returns:
            Tuple[AudioDiffusionDataset, AudioDiffusionDataset]: A tuple containing the training and testing datasets.

        Raises:
            RuntimeError: If the dataset name is unknown.
        """
        transform = transforms.Compose([transforms.ToTensor()])

        # Check if resizing is required
        if name == Case.audio_diffusion_64:
            transform.transforms.append(transforms.Resize((64, 64)))

        transform.transforms.append(transforms.Lambda(AudioDataUtils.scale_imgs))

        if name in [Case.audio_diffusion_256, Case.audio_diffusion_64]:
            dataset = load_dataset("teticio/audio-diffusion-256")["train"]
            train_size = int(0.9 * len(dataset))
            test_size = len(dataset) - train_size

            generator = torch.Generator().manual_seed(42)
            train_dataset, test_dataset = random_split(
                dataset, [train_size, test_size], generator=generator
            )
        else:
            raise RuntimeError(f"Unkwown audio dataset {name}")

        audio_train_dataset = AudioDiffusionDataset(train_dataset, transform=transform)
        audio_test_dataset = AudioDiffusionDataset(test_dataset, transform=transform)

        return audio_train_dataset, audio_test_dataset

    @staticmethod
    def get_mel(data_type: str):
        """
        Gets the mel spectrogram configuration for the specified data type.

        Parameters:
            data_type (str): The type of audio data.

        Returns:
            Mel: An instance of the Mel class with configured parameters.

        Raises:
            RuntimeError: If the data type is unknown.
        """
        if data_type == Case.audio_diffusion_256:
            x_res, y_res = 256, 256
            hop_length = 512
        elif data_type == Case.audio_diffusion_64:
            x_res, y_res = 64, 64
            hop_length = 1024
        else:
            raise RuntimeError(f"Unkown audio data type {data_type}")

        mel = Mel(
            x_res=x_res,
            y_res=y_res,
            hop_length=hop_length,
            sample_rate=22050,
            n_fft=2048,
            n_iter=512,
        )

        return mel

    @staticmethod
    def scale_imgs(t: torch.Tensor) -> torch.Tensor:
        """
        Scales the input tensor.

        Parameters:
        - t (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Scaled tensor.
        """
        return (t * 2) - 1
