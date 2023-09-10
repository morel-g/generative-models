import torch
from sklearn.model_selection import train_test_split

# from datasets import load_dataset, Features, Array2D
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from ..distribution_toy import inf_train_gen
from .data_type import toy_data_type
from datasets import load_dataset
from torchvision.datasets import CelebA, CIFAR10, FashionMNIST, MNIST
from ..case import Case


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, x, y=None, transform=None):
        "Initialization"
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        len_x = len(self.x) if self.x is not None else 0
        len_y = len(self.y) if self.y is not None else 0
        return max(len_x, len_y)

    def __getitem__(self, index):
        "Generates one sample of data"
        x_idx = self.x[index] if self.x is not None else None

        if self.y is not None:
            if self.transform:
                return self.transform(x_idx), self.transform(self.y[index])
            return x_idx, self.y[index]
        else:
            if self.transform:
                return self.transform(x_idx)
            return x_idx


def compute_mean_and_std(data_type):
    dataset = prepare_dataset(data_type)[0]
    loader = DataLoader(dataset, batch_size=len(dataset) // 100)
    mean, std = None, None
    nb_samples = 0
    for xi in loader:
        xi = xi[0] if len(xi) == 2 else xi
        if mean is None:
            mean, std = xi.mean((0, 2, 3)), xi.std((0, 2, 3))
        else:
            mean += xi.mean((0, 2, 3))
            std += xi.std((0, 2, 3))
        nb_samples += 1
    return mean / nb_samples, std / nb_samples


def get_dataset(data_type, log_dir, normalized_img=True, n_samples=None):
    if data_type in toy_data_type:
        if n_samples in (0, None):
            raise RuntimeError(
                "For 2d dataset the number of samples should be an integer > 0."
            )
        return prepare_toy_dataset(data_type, n_samples, log_dir)
    else:
        if not normalized_img:
            return prepare_dataset(data_type)
        else:
            mean_std = compute_mean_and_std(data_type)
            return prepare_dataset(data_type, mean_std=mean_std)


# def prepare_dataset(dataset_name="fashion_mnist"):
#     # load dataset from the hub
#     dataset = load_dataset(dataset_name)
#     # define image transformations (e.g. using torchvision)
#     transform = Compose(
#         [
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda t: (t * 2) - 1),
#         ]
#     )

#     # define function
#     def transforms_func(examples):
#         examples["pixel_values"] = [
#             transform(image.convert("L")) for image in examples["image"]
#         ]
#         del examples["image"]
#         return examples


#     # dataset = dataset.map(
#     #     transforms_func, remove_columns=["image"], batched=True
#     # ).remove_columns("label")
#     transformed_dataset = dataset.with_transform(
#         transforms_func
#     ).remove_columns("label")
#     train_data, val_data = (
#         transformed_dataset["train"],
#         transformed_dataset["test"],
#     )
#     # features = Features({"data": Array2D(shape=(28, 28), dtype="float32")})
#     return (
#         train_data,
#         val_data,  # .set_format(type="torch")
#     )
def scale_imgs(t):
    return (t * 2) - 1


def prepare_dataset(name, mean_std=None):
    transform = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(scale_imgs),
    ]
    if name == Case.cifar10_grayscale:
        transform.append(transforms.Grayscale())
    if mean_std is not None:
        transform.append(transforms.Normalize(*mean_std))

    transform = Compose(transform)

    if name in (Case.cifar10, Case.cifar10_grayscale):
        train_data = CIFAR10(
            "~/datasets",
            train=True,
            transform=transform,
            download=True,
        )
        val_data = CIFAR10(
            "~/datasets",
            train=False,
            transform=transform,
            download=True,
        )
    elif name == Case.fashion_mnist:
        train_data = FashionMNIST(
            "~/datasets",
            train=True,
            transform=transform,
            download=True,
        )
        val_data = FashionMNIST(
            "~/datasets",
            train=False,
            transform=transform,
            download=True,
        )
    elif name == Case.mnist:
        train_data = MNIST(
            "~/datasets",
            train=True,
            transform=transform,
            download=True,
        )
        val_data = MNIST(
            "~/datasets",
            train=False,
            transform=transform,
            download=True,
        )
    else:
        raise RuntimeError("Unkown dataset.")
    # train_data = CelebA(
    #     "/home/morel/datasets",
    #     split="train",
    #     transform=transform,
    #     download=True,
    # )
    # val_data = CelebA(
    #     "/home/morel/datasets",
    #     split="val",
    #     transform=transform,
    #     download=True,
    # )
    # train_data = datasets.MNIST(
    #     "datasets", train=True, download=True, transform=transform
    # )
    # val_data = datasets.MNIST("datasets", train=False, transform=transform)
    return train_data, val_data


def prepare_toy_data(data_type, n_samples, path="outputs"):
    X = inf_train_gen(data_type, batch_size=n_samples, path=path)
    X = torch.tensor(X, dtype=torch.float32)
    x_y = list(train_test_split(X, test_size=0.20, random_state=42))

    return x_y


def prepare_toy_dataset(data_type, n_samples, log_dir):
    """Load / construct the dataset.

    Args:
        data: A Data object.
        logger: Logger of the simulation.

    Raises:
        RuntimeError: Unknown data.data_type.

    Returns:
        A tuple made of a training and a validation dataset.
    """

    x_train, x_val = prepare_toy_data(data_type, n_samples, path=log_dir)
    x_train, x_val = Dataset(x_train), Dataset(x_val)

    return (x_train, x_val)


def get_fashion_mnist_dataloader():
    batch_size = 128

    train_data, _ = prepare_dataset("fashion_mnist")

    # create dataloader
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return dataloader


def get_celeba_dataloader():
    batch_size = 128

    train_data, _ = prepare_dataset("fashion_mnist")

    # create dataloader
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return dataloader
