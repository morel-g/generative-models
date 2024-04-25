import os
import torch
import numpy as np
from src.data_manager.dataset import Dataset
from src.utils import split_train_test


class CustomDataUtils:
    @staticmethod
    def load_and_concatenate(dir_path, prefix):
        files = [f for f in os.listdir(dir_path) if f.startswith(prefix)]
        files.sort()

        arrays = [np.load(os.path.join(dir_path, f)) for f in files]
        concatenated_array = np.concatenate(arrays, axis=0)
        return torch.tensor(concatenated_array)

    @staticmethod
    def prepare_custom_dataset(
        data_dir, prefix_1, prefix_2=None
    ) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
        x1 = CustomDataUtils.load_and_concatenate(data_dir, prefix_1)
        if prefix_2 is not None:
            x2 = CustomDataUtils.load_and_concatenate(data_dir, prefix_2)
            x_train, x_test = split_train_test(torch.cat((x1, x2), dim=1))
            x1_train, x2_train = torch.chunk(x_train, 2, dim=1)
            x1_test, x2_test = torch.chunk(x_test, 2, dim=1)
        else:
            x1_train, x1_test = split_train_test(x1)
            x2_train, x2_test = None, None

        return Dataset(x1_train, x2_train), Dataset(x1_test, x2_test)
