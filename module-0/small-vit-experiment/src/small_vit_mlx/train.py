from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from .data import iterate_minibatches, load_cifar10, make_synthetic_cifar, normalize_images
from .model import SmallViT, count_parameters


def accuracy(logits: mx.array, labels: mx.array) -> mx.array:
    predictions = mx.argmax(logits, axis=-1)
    return mx.mean(predictions == labels)


def evaluate(model: SmallViT, images: np.ndarray, labels: np.ndarray, batch_size: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    for batch_images, batch_labels in iterate_minibatches(images, labels, batch_size, shuffle=False):
        x = mx.array(batch_images)
        y = mx.array(batch_labels)
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        acc = accuracy(logits, y)
        mx.eval(loss, acc)

        batch_count = len(batch_images)
        total_loss += float(loss) * batch_count
        total_acc += float(acc) * batch_count
        total_seen += batch_count

    model.train()
    return total_loss / total_seen, total_acc / total_seen


def train(args: argparse.Namespace) -> None:
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke_test:
        (train_images, train_labels), (test_images, test_labels) = make_synthetic_cifar()
    else:
        (train_images, train_labels), (test_images, test_labels) = load_cifar10(args.data_dir)
        train_images = normalize_images(train_images)
        test_images = normalize_images(test_images)

    if args.train_limit:
        train_images = train_images[: args.train_limit]
        train_labels = train_labels[: args.train_limit]
    if args.test_limit:
        test_images = test_images[: args.test_limit]
        test_labels = test_labels[: args.test_limit]

    model = SmallViT(
        image_size=32,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parameters: {count_parameters(model):,}")
    print(f"Train examples: {len(train_images):,} | Test examples: {len(test_images):,}")

    def loss_fn(model: SmallViT, x: mx.array, y: mx.array) -> mx.array:
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_seen = 0

        progress = tqdm(
            iterate_minibatches(train_images, train_labels, args.batch_size, shuffle=True),
            total=(len(train_images) + args.batch_size - 1) // args.batch_size,
            desc=f"epoch {epoch}/{args.epochs}",
        )
        for batch_images, batch_labels in progress:
            x = mx.array(batch_images)
            y = mx.array(batch_labels)

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)

            logits = model(x)
            acc = accuracy(logits, y)
            mx.eval(loss, acc, model.parameters(), optimizer.state)

            batch_count = len(batch_images)
            total_loss += float(loss) * batch_count
            total_acc += float(acc) * batch_count
            total_seen += batch_count
            progress.set_postfix(
                loss=f"{total_loss / total_seen:.4f}",
                acc=f"{total_acc / total_seen:.3f}",
            )

        test_loss, test_acc = evaluate(model, test_images, test_labels, args.batch_size)
        print(
            f"epoch {epoch}: "
            f"train_loss={total_loss / total_seen:.4f} "
            f"train_acc={total_acc / total_seen:.3f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.3f}"
        )

        latest_path = args.checkpoint_dir / "latest.safetensors"
        model.save_weights(str(latest_path))
        if args.save_every and epoch % args.save_every == 0:
            model.save_weights(str(args.checkpoint_dir / f"epoch-{epoch}.safetensors"))
        print(f"saved checkpoint: {latest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small MLX Vision Transformer on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
