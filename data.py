import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def load_data(batch_size: int, ratio: float = 0.8):
    # load the MNIST dataset
    train_dataset = MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root="data", train=False, download=True, transform=ToTensor())

    train_size = int(ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(
        (
            f"train_loader: {len(train_loader.dataset)}, val_loader: {len(val_loader.dataset)}, "
            f"test_loader: {len(test_loader.dataset)}"
        )
    )
    return train_loader, val_loader, test_loader


def slice_data(data_loader, batch_size: int, ratio: float):
    sample_len = int(len(data_loader.dataset) * ratio)
    dataset, _ = torch.utils.data.random_split(
        data_loader.dataset, [sample_len, len(data_loader.dataset) - sample_len]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
