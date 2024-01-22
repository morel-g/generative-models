import os
import torch
import pandas as pd
import subprocess
from src.data_manager.dataset import Dataset
from typing import Tuple

from src.case import Case
from src.utils import split_train_test


class ManifoldDataUtils:
    manifold_dir = "data/manifold"

    @staticmethod
    def prepare_manifold_dataset(data_type: str) -> Tuple[Dataset, Dataset]:
        ManifoldDataUtils.download_manifold_data()
        if data_type == Case.earthquake:
            filename = os.path.join(
                ManifoldDataUtils.manifold_dir, "earth_data/earthquake.csv"
            )
        elif data_type == Case.fire:
            filename = os.path.join(
                ManifoldDataUtils.manifold_dir, "earth_data/fire.csv"
            )
        elif data_type == Case.flood:
            filename = os.path.join(
                ManifoldDataUtils.manifold_dir, "earth_data/flood.csv"
            )
        df = pd.read_csv(filename)
        dataset = torch.tensor(df.to_numpy()).float()
        dataset = ManifoldDataUtils.cartesian_from_latlon(dataset)

        train_dataset, test_dataset = split_train_test(dataset)

        return Dataset(train_dataset), Dataset(test_dataset)

    @staticmethod
    def download_manifold_data():
        # Define the directories and file path
        data_dir = "data"
        manifold_dir = ManifoldDataUtils.manifold_dir
        zip_url = "https://rtqichen.com/manifold_data/data.zip"
        zip_file = "data.zip"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if not os.path.exists(manifold_dir):
            subprocess.run(["wget", zip_url])
            subprocess.run(["unzip", "-q", zip_file, "-d", data_dir])
            os.rename("data/data", manifold_dir)
            os.remove(zip_file)

    @staticmethod
    def cartesian_from_latlon(x):
        assert x.shape[-1] == 2
        x = x * torch.pi / 180
        lat = x.select(-1, 0)
        lon = x.select(-1, 1)
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

    @staticmethod
    def latlon_from_cartesian(x):
        r = x.pow(2).sum(-1).sqrt()
        x, y, z = x[..., 0], x[..., 1], x[..., 2]
        lat = torch.asin(z / r)
        lon = torch.atan2(y, x)
        return (
            torch.cat([lat.unsqueeze(-1), lon.unsqueeze(-1)], dim=-1) * 180 / torch.pi
        )
