import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

from src.case import Case


class ImgDataUtils:
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

    @staticmethod
    def prepare_img_dataset(
        name: str, mean_std: tuple = None
    ) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
        """
        Prepares a dataset based on the given name and normalization parameters.

        Parameters:
        - name (str): Name of the dataset.
        - mean_std (tuple, optional): Mean and standard deviation for normalization.

        Returns:
        - Tuple[torch.utils.data.Dataset]: Training and validation datasets.
        """
        transform = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(ImgDataUtils.scale_imgs),
        ]

        if name == Case.cifar10_grayscale:
            transform.append(transforms.Grayscale())
        if mean_std is not None:
            transform.append(transforms.Normalize(*mean_std))

        transform = Compose(transform)

        dataset_map = {
            Case.cifar10: CIFAR10,
            Case.cifar10_grayscale: CIFAR10,
            Case.fashion_mnist: FashionMNIST,
            Case.mnist: MNIST,
        }

        if name in dataset_map:
            train_data = dataset_map[name](
                "~/datasets", train=True, transform=transform, download=True
            )
            val_data = dataset_map[name](
                "~/datasets", train=False, transform=transform, download=True
            )
        else:
            raise RuntimeError("Unknown dataset.")

        return train_data, val_data


# def compute_mean_and_std(data_type: str) -> (torch.Tensor, torch.Tensor):
#     """
#     Computes the mean and standard deviation for a given data type.

#     Parameters:
#     - data_type (str): Type of the data.

#     Returns:
#     - Tuple[torch.Tensor]: Mean and standard deviation tensors.
#     """
#     dataset = prepare_img_dataset(data_type)[0]
#     loader = DataLoader(dataset, batch_size=len(dataset) // 100)
#     mean, std = None, None
#     nb_samples = 0
#     for xi in loader:
#         xi = xi[0] if len(xi) == 2 else xi
#         if mean is None:
#             mean, std = xi.mean((0, 2, 3)), xi.std((0, 2, 3))
#         else:
#             mean += xi.mean((0, 2, 3))
#             std += xi.std((0, 2, 3))
#         nb_samples += 1
#     return mean / nb_samples, std / nb_samples
