# <Source: https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py >

import numpy as np
import sklearn
import torch
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_swiss_roll
import os


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200, path=None):
    if rng is None:
        rng = np.random.RandomState()
        # print(rng)

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size, noise=0.0  # 1.0
        )[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(
            n_samples=batch_size, factor=0.5, noise=0.08
        )[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "moons":
        data, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)

        if path is not None:
            # Save the labels.
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            _, y = list(train_test_split(y, test_size=0.20, random_state=42))
            np.save(path + "/class_points.npy", y)

        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        data = data.astype("float32")

        # normalize data
        data = data - np.mean(data, 0)
        # data = data / np.sqrt(np.var(data, axis=0))
        data = data / (0.5 * np.sqrt((data**2).mean(0)))
        # data = data / (np.sqrt((data**2).mean(0)))

        return data
    elif data == "eight_gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        labels = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
            labels.append(idx)

        if path is not None:
            # Save the labels.
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            _, labels = list(
                train_test_split(labels, test_size=0.20, random_state=42)
            )
            np.save(path + "/class_points.npy", labels)

        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414

        # normalize data
        # dataset = dataset - np.mean(dataset, 0)
        # dataset = dataset / (0.5 * np.sqrt((dataset**2).mean(0)))

        return dataset
    elif data == "gaussian":
        return rng.randn(batch_size, 2)
    elif data == "conditionnal8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        context = np.zeros((batch_size, 8))
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            context[i, idx] = 1
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, context

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes * num_per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(
            np.einsum("ti,tij->tj", features, rotations)
        ).astype("float32")

    elif data == "2spirals":
        n = (
            np.sqrt(np.random.rand(batch_size // 2, 1))
            * 540
            * (2 * np.pi)
            / 360
        )
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x.astype("float32")

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = (
            np.random.rand(batch_size)
            - np.random.randint(0, 2, batch_size) * 2
        )
        x2 = x2_ + (np.floor(x1) % 2)
        return (
            np.concatenate([x1[:, None], x2[:, None]], 1).astype("float32") * 2
        )

    elif data == "line":
        x = rng.rand(batch_size)
        # x = np.arange(0., 1., 1/batch_size)
        x = x * 5 - 2.5
        y = x  # - x + rng.rand(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "line-noisy":
        x = rng.rand(batch_size)
        x = x * 5 - 2.5
        y = x + rng.randn(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1).astype("float32")
    elif data == "joint_gaussian":
        x2 = torch.distributions.Normal(0.0, 4.0).sample((batch_size, 1))
        x1 = (
            torch.distributions.Normal(0.0, 1.0).sample((batch_size, 1))
            + (x2**2) / 4
            - 4.0
        )  # E[X^2]=sigma^2

        return torch.cat((x1, x2), 1)
    elif data == "uniform":
        return torch.rand(batch_size, 2) - 0.5
    elif data == "multimodal_swissroll":
        NOISE = 0.2
        MULTIPLIER = 0.01
        OFFSETS = [[0.8, 0.8], [0.8, -0.8], [-0.8, -0.8], [-0.8, 0.8]]

        idx = np.random.multinomial(batch_size, [0.2] * 5, size=1)[0]

        sr = []
        for k in range(5):
            sr.append(
                make_swiss_roll(int(idx[k]), noise=NOISE)[0][:, [0, 2]].astype(
                    "float32"
                )
                * MULTIPLIER
            )

            if k > 0:
                sr[k] += np.array(OFFSETS[k - 1]).reshape(-1, 2)

        data = np.concatenate(sr, axis=0)[np.random.permutation(batch_size)]

        # normalize data
        # data = data - np.mean(data, 0)
        # data = data / (np.sqrt((data**2).mean(0)))

        return data
    else:
        raise RuntimeError("Unknwon distribution")
