from torchvision import datasets, transforms
from typing import Optional, List, Dict, Union, Tuple, Any
from torch.utils.data import Subset, random_split, DataLoader, Dataset
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set torch, cuda and numpy seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def check_labels(dataloader: DataLoader) -> None:
    """
    Function to check the labels of the dataset.
    Args:
        dataloader: torch dataloader
    """

    label_counts = defaultdict(int)

    for images, labels in dataloader:
        for label in labels:
            label_counts[label.item()] += 1

    for label, count in label_counts.items():
        print(f"Label {label}: {count}")


def plot_image(dataloader: DataLoader, selected_label: int) -> None:
    """
    Simple function to print the image on the screen given a label.

    Args:
        dataloader: torch dataloader
        selected_label: label to be selected
    """
    for images, labels in dataloader:
        image = images[0]
        label = labels[0]
        if label == selected_label:
            image_np = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            plt.imshow(image_np)
            plt.title(f'Label: {label.item()}, Size: {image_np.shape}')
            plt.axis('off')
            plt.show()
            break


def process_mnist(
        resize: Optional[int] = 28,
        train_validation_split: Optional[int] = None,
        test_validation_split: Optional[int] = None,
        batch_size: int = 64,
        selected_labels: Optional[List[int]] = None,
        num_workers: int = 0,
        train_samples: Optional[int] = None,
        test_samples: Optional[int] = None
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Loads and processes the MNIST dataset with optional resizing, filtering, and splitting.

    Args:
        resize: Resize dimension for images (square).
        train_validation_split: Percentage of training data to use as validation.
        test_validation_split: Percentage of test data to use as validation.
        batch_size: Number of samples per batch.
        selected_labels: List of digit labels to include. If None, all digits are used.
        num_workers: Number of subprocesses for data loading.
        train_samples: Limit number of samples from training set.
        test_samples: Limit number of samples from test set.

    Returns:
        Tuple containing (train_loader, validation_loader or None, test_loader)
    """

    def load_dataset(train: bool) -> datasets.MNIST:
        return datasets.MNIST(
            root='./data',
            download=True,
            train=train,
            transform=transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor()
            ])
        )

    def filter_labels(dataset, labels: List[int]) -> Subset:
        indices = [i for i, label in enumerate(dataset.targets) if label.item() in labels]
        return Subset(dataset, indices)

    def limit_samples(dataset, max_samples: Optional[int]) -> Subset:
        if max_samples is None:
            return dataset
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))

    def create_loader(dataset, shuffle=True) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=True)

    # Load datasets
    train_dataset = load_dataset(train=True)
    test_dataset = load_dataset(train=False)

    # Filter labels if needed
    if selected_labels is not None:
        train_dataset = filter_labels(train_dataset, selected_labels)
        test_dataset = filter_labels(test_dataset, selected_labels)

    # Limit number of samples
    train_dataset = limit_samples(train_dataset, train_samples)
    test_dataset = limit_samples(test_dataset, test_samples)

    # Handle split logic
    if train_validation_split is not None:
        val_size = int(len(train_dataset) * train_validation_split / 100)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        return (
            create_loader(train_subset),
            create_loader(val_subset),
            create_loader(test_dataset)
        )

    if test_validation_split is not None:
        val_size = int(len(test_dataset) * test_validation_split / 100)
        test_size = len(test_dataset) - val_size
        test_subset, val_subset = random_split(test_dataset, [test_size, val_size])
        return (
            create_loader(train_dataset),
            create_loader(val_subset),
            create_loader(test_subset)
        )

    return create_loader(train_dataset), None, create_loader(test_dataset)
