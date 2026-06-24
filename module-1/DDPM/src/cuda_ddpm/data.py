import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OXFORD_FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def list_images(data_dir: str) -> list[Path]:
    root = Path(data_dir)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise FileNotFoundError(f"No images found in {root}")
    return sorted(paths)


def download_oxford_flowers(data_dir: str = "data/oxford-flowers") -> Path:
    """Download and extract Oxford Flowers 102 if it is not already present."""
    root = Path(data_dir)
    image_dir = root / "jpg"
    if image_dir.exists() and any(image_dir.glob("*.jpg")):
        return image_dir

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "102flowers.tgz"
    if not archive_path.exists():
        _download_file(OXFORD_FLOWERS_URL, archive_path)

    print(f"Extracting Oxford Flowers 102 into {root}...")
    _safe_extract_tar(archive_path, root)

    if not image_dir.exists() or not any(image_dir.glob("*.jpg")):
        raise RuntimeError(f"Oxford Flowers extraction did not create images in {image_dir}")
    return image_dir


def download_cifar10(data_dir: str = "data/cifar-10") -> Path:
    """Download CIFAR-10 and convert the Python batches to PNG images."""
    root = Path(data_dir)
    image_dir = root / "images"
    if image_dir.exists() and any(image_dir.glob("*.png")):
        return image_dir

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "cifar-10-python.tar.gz"
    if not archive_path.exists():
        _download_file(CIFAR10_URL, archive_path, "CIFAR-10")

    extracted_dir = root / "cifar-10-batches-py"
    if not extracted_dir.exists():
        print(f"Extracting CIFAR-10 into {root}...")
        _safe_extract_tar(archive_path, root)

    print(f"Converting CIFAR-10 batches to PNG images in {image_dir}...")
    image_dir.mkdir(parents=True, exist_ok=True)
    for batch_index in range(1, 6):
        _convert_cifar_batch(extracted_dir / f"data_batch_{batch_index}", image_dir, f"train_{batch_index}")
    _convert_cifar_batch(extracted_dir / "test_batch", image_dir, "test")

    if not any(image_dir.glob("*.png")):
        raise RuntimeError(f"CIFAR-10 conversion did not create images in {image_dir}")
    return image_dir


def _download_file(url: str, destination: Path, name: str = "Oxford Flowers 102") -> None:
    temp_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {name} from {url}")
    print(f"Destination: {destination}")

    request = urllib.request.Request(url, headers={"User-Agent": "cuda-ddpm/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        total = int(response.headers.get("Content-Length", 0))
        with temp_path.open("wb") as file:
            with tqdm(total=total or None, unit="B", unit_scale=True, desc="download") as progress:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress.update(len(chunk))

    temp_path.replace(destination)


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path) as archive:
        destination_resolved = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not (target == destination_resolved or destination_resolved in target.parents):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def _convert_cifar_batch(batch_path: Path, image_dir: Path, prefix: str) -> None:
    with batch_path.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")

    data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b"labels"]
    for index, image in enumerate(data):
        path = image_dir / f"{prefix}_{index:05d}_class_{labels[index]}.png"
        if not path.exists():
            Image.fromarray(image).save(path)


def load_image(path: Path, image_size: int, channels: int) -> np.ndarray:
    mode = "L" if channels == 1 else "RGB"
    image = Image.open(path).convert(mode)
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    if channels == 1:
        array = array[..., None]
    return array.transpose(2, 0, 1)


def image_batches(paths: list[Path], batch_size: int, image_size: int, channels: int):
    order = np.arange(len(paths))
    while True:
        np.random.shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if len(indices) != batch_size:
                continue
            yield np.stack([load_image(paths[i], image_size, channels) for i in indices])
