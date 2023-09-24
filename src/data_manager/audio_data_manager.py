import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from datasets import load_dataset
from diffusers import Mel

from src.case import Case


class AudioDiffusionDataset(Dataset):
    def __init__(self, huggingface_dataset, transform=None):
        self.huggingface_dataset = huggingface_dataset
        self.transform = transform

    def __len__(self):
        return len(self.huggingface_dataset)

    def __getitem__(self, idx):
        image = self.huggingface_dataset[idx]["image"]
        audio_path = self.huggingface_dataset[idx]["audio_file"]

        if self.transform:
            image = self.transform(image)

        return image, audio_path


def get_mel(data_type):
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


def scale_imgs(t: torch.Tensor) -> torch.Tensor:
    """
    Scales the input tensor.

    Parameters:
    - t (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Scaled tensor.
    """
    return (t * 2) - 1


def load_audio_dataset(name):
    transform = transforms.Compose([transforms.ToTensor()])

    # Check if resizing is required
    if name == Case.audio_diffusion_64:
        transform.transforms.append(transforms.Resize((64, 64)))

    transform.transforms.append(transforms.Lambda(scale_imgs))

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

    audio_train_dataset = AudioDiffusionDataset(
        train_dataset, transform=transform
    )
    audio_test_dataset = AudioDiffusionDataset(
        test_dataset, transform=transform
    )

    return audio_train_dataset, audio_test_dataset
