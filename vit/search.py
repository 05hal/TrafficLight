#!/usr/bin/env python
"""Hyperparameter search for ViT traffic-light classification.

Runs a grid search over the defined space, reports results, and optionally
retrains with the best configuration on the full training set.
"""

import argparse
import csv
import itertools
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from train_vit import TrainConfig, TrainResult, build_model, build_transforms, get_processor, plot_curves, run_epoch, train


SEARCH_SPACE = {
    "lr": [1e-5, 2e-5, 5e-5],
    "epochs": [30, 50],
    "weight_decay": [1e-4, 1e-3],
    "aug_strength": [0.2, 0.4],
}


def generate_configs(space: dict, data_dir: str, batch_size: int, num_workers: int) -> list[dict]:
    keys = list(space.keys())
    configs = []
    for values in itertools.product(*[space[k] for k in keys]):
        cfg = dict(zip(keys, values))
        cfg["data_dir"] = data_dir
        cfg["batch_size"] = batch_size
        cfg["num_workers"] = num_workers
        configs.append(cfg)
    return configs


def plot_search_results(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = []
    accs = []
    for i, r in enumerate(results):
        label = f"lr={r['lr']:.0e}\nep={r['epochs']}\nwd={r['weight_decay']:.0e}\naug={r['aug_strength']}"
        labels.append(label)
        accs.append(r["best_val_acc"])

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(accs)))
    bars = ax.bar(range(len(accs)), accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Best Val Accuracy")
    ax.set_title("Hyperparameter Search Results (ViT)")
    ax.set_ylim(0, 1.05)

    best_idx = int(np.argmax(accs))
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        weight = "bold" if i == best_idx else "normal"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight=weight)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Search results chart: {out_path}")


def retrain_best(best_cfg: dict, data_dir: str, vit_dir: Path) -> None:
    """Retrain with best config on combined train+val for final model."""
    print("\n" + "=" * 60)
    print("Retraining with best config on full train+val set...")
    print("=" * 60)

    # merge train+val into a temp directory
    merged_dir = vit_dir / "_merged_dataset"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    for split in ["train", "val"]:
        src = Path(data_dir) / split
        if not src.exists():
            continue
        for cls_dir in src.iterdir():
            if cls_dir.is_dir():
                dst = merged_dir / "train" / cls_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in cls_dir.iterdir():
                    shutil.copy2(f, dst / f.name)

    # use original val as val (for monitoring)
    val_src = Path(data_dir) / "val"
    if val_src.exists():
        for cls_dir in val_src.iterdir():
            if cls_dir.is_dir():
                dst = merged_dir / "val" / cls_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in cls_dir.iterdir():
                    shutil.copy2(f, dst / f.name)

    cfg = TrainConfig(
        data_dir=str(merged_dir),
        epochs=best_cfg["epochs"],
        batch_size=best_cfg["batch_size"],
        lr=best_cfg["lr"],
        weight_decay=best_cfg["weight_decay"],
        aug_strength=best_cfg["aug_strength"],
        num_workers=best_cfg["num_workers"],
        use_amp=True,
        checkpoint_dir=str(vit_dir / "checkpoints"),
        results_dir=str(vit_dir / "results"),
    )

    result = train(cfg)
    print(f"\nFinal model: val_acc={result.best_val_acc:.3f}, epoch={result.best_epoch}")
    print(f"Checkpoint: {result.checkpoint_path}")

    if merged_dir.exists():
        shutil.rmtree(merged_dir)


def main() -> None:
    vit_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Hyperparameter search for ViT.")
    parser.add_argument("--data", type=Path, default=vit_dir / "dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (use max your GPU allows).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--search-dir", type=Path, default=vit_dir / "checkpoints" / "search")
    parser.add_argument("--retrain", action="store_true", help="Retrain best config on merged train+val.")
    args = parser.parse_args()

    args.search_dir.mkdir(parents=True, exist_ok=True)
    data_dir = str(args.data)

    configs = generate_configs(SEARCH_SPACE, data_dir, args.batch_size, args.num_workers)
    print(f"Search space: {len(configs)} configurations")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # CSV header
    csv_path = args.search_dir / "search_results.csv"
    fieldnames = ["run", "lr", "epochs", "weight_decay", "aug_strength", "best_val_acc", "best_epoch", "elapsed"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    results = []
    t_total = time.time()

    for i, cfg_dict in enumerate(configs):
        print(f"\n--- Run {i + 1}/{len(configs)} ---")
        print(f"  lr={cfg_dict['lr']:.0e}  epochs={cfg_dict['epochs']}  "
              f"wd={cfg_dict['weight_decay']:.0e}  aug={cfg_dict['aug_strength']}")

        run_ckpt_dir = args.search_dir / f"run_{i + 1:02d}"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        cfg = TrainConfig(
            data_dir=cfg_dict["data_dir"],
            epochs=cfg_dict["epochs"],
            batch_size=cfg_dict["batch_size"],
            lr=cfg_dict["lr"],
            weight_decay=cfg_dict["weight_decay"],
            aug_strength=cfg_dict["aug_strength"],
            num_workers=cfg_dict["num_workers"],
            use_amp=True,
            checkpoint_dir=str(run_ckpt_dir),
            results_dir="",
            verbose=False,
        )

        result = train(cfg)
        elapsed = result.elapsed

        row = {
            "run": i + 1,
            "lr": cfg_dict["lr"],
            "epochs": cfg_dict["epochs"],
            "weight_decay": cfg_dict["weight_decay"],
            "aug_strength": cfg_dict["aug_strength"],
            "best_val_acc": f"{result.best_val_acc:.4f}",
            "best_epoch": result.best_epoch,
            "elapsed": f"{elapsed:.1f}",
        }
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        cfg_dict["best_val_acc"] = result.best_val_acc
        cfg_dict["best_epoch"] = result.best_epoch
        results.append(cfg_dict)

        print(f"  => val_acc={result.best_val_acc:.3f} (epoch {result.best_epoch}) in {elapsed:.1f}s")

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Search complete: {len(results)} runs in {total_time:.1f}s")
    print(f"{'=' * 60}")

    # find best
    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"\nBest config:")
    print(f"  lr={best['lr']:.0e}  epochs={best['epochs']}  "
          f"wd={best['weight_decay']:.0e}  aug={best['aug_strength']}")
    print(f"  val_acc={best['best_val_acc']:.3f}")

    # save best config
    best_path = args.search_dir / "best_config.json"
    best_path.write_text(json.dumps({k: v for k, v in best.items()}, indent=2, default=str), encoding="utf-8")
    print(f"  saved: {best_path}")

    # plot
    plot_search_results(results, args.search_dir / "search_results.png")
    print(f"  CSV: {csv_path}")

    if args.retrain:
        retrain_best(best, data_dir, vit_dir)


if __name__ == "__main__":
    main()
