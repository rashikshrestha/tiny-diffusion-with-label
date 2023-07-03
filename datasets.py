import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def all_dataset(n=8000):
    dnames = ['moons', 'dino', 'line', 'circle']

    dataset_acc = []
    for idx, name in enumerate(dnames):
        dataset = get_dataset(name)

        for d in dataset:
            data_point = [d[0][0].item(), d[0][1].item(), idx]
            dataset_acc.append(data_point)

    all_dataset = torch.tensor(dataset_acc)

    return TensorDataset(all_dataset)


def point_dataset(point_data, n=8000):
    point_data = [point_data,]
    return torch.tensor(point_data*n)

def all_point_dataset(n=8000):
    points = [[2,2,0], [2,-2,1], [-2,-2,2], [-2,2,3]]

    all_dataset = []
    for p in points:
        all_dataset.append(point_dataset(p, n))

    all_dataset = torch.cat(all_dataset)

    return TensorDataset(all_dataset)

# data = all_point_dataset()
# print(data.shape)
# print(data[0])
# print(data[7999])
# print(data[8000])
# print(data[16000])
# print(data[2400])



