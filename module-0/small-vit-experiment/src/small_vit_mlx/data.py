from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def download_cifar10(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir / "cifar-10-batches-py"
    if extracted_dir.exists():
        return extracted_dir

    archive_path = data_dir / "cifar-10-python.tar.gz"
    if not archive_path.exists():
        print(f"Downloading CIFAR-10 to {archive_path}...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)

    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(data_dir)
    return extracted_dir


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")

    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch["labels"], dtype=np.int64)
    return images, labels


def load_cifar10(data_dir: Path) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    cifar_dir = download_cifar10(data_dir)
    train_images = []
    train_labels = []
    for index in range(1, 6):
        images, labels = _load_batch(cifar_dir / f"data_batch_{index}")
        train_images.append(images)
        train_labels.append(labels)

    test_images, test_labels = _load_batch(cifar_dir / "test_batch")
    return (
        np.concatenate(train_images).astype(np.float32),
        np.concatenate(train_labels),
    ), (test_images.astype(np.float32), test_labels)


def normalize_images(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32) / 255.0
    return (images - CIFAR10_MEAN) / CIFAR10_STD


def make_synthetic_cifar(num_train: int = 512, num_test: int = 128):
    rng = np.random.default_rng(0)
    train_images = rng.normal(size=(num_train, 32, 32, 3)).astype(np.float32)
    train_labels = rng.integers(0, 10, size=(num_train,), dtype=np.int64)
    test_images = rng.normal(size=(num_test, 32, 32, 3)).astype(np.float32)
    test_labels = rng.integers(0, 10, size=(num_test,), dtype=np.int64)
    return (train_images, train_labels), (test_images, test_labels)


def iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]
