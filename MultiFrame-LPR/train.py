#!/usr/bin/env python3
"""Main entry point for the multi-frame LPR training pipeline."""
import argparse
import os
import shutil
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path for imports.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.data.paths import find_named_test_root, find_train_root, has_tracks
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition",
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/submission files.",
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran"], default=None,
        help="Model architecture: crnn or restran.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training.")
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None, dest="learning_rate",
        help="Learning rate.",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data. If omitted, local/Kaggle layouts are auto-detected.",
    )
    parser.add_argument(
        "--test-data-root", type=str, default=None,
        help="Generic test root for submission mode.",
    )
    parser.add_argument(
        "--public-test-root", type=str, default=None,
        help="Kaggle public test root, for example Pa7a3Hin-test-public.",
    )
    parser.add_argument(
        "--blind-test-root", type=str, default=None,
        help="Kaggle blind test root, for example TKzFBtn7-test-blind.",
    )
    parser.add_argument(
        "--val-split-file", type=str, default=None,
        help="Validation split JSON file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--num-frames", type=int, default=None, help="Frames per track.")
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="LSTM hidden size for CRNN.",
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=None,
        help="Number of transformer attention heads.",
    )
    parser.add_argument(
        "--transformer-layers", type=int, default=None,
        help="Number of transformer encoder layers.",
    )
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level for training data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and submission files.",
    )
    parser.add_argument("--use-stn", action="store_true", help="Enable STN alignment.")
    parser.add_argument("--no-stn", action="store_true", help="Disable STN alignment.")
    parser.add_argument("--use-amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument(
        "--force-single-gpu",
        action="store_true",
        help="Disable DataParallel even when multiple GPUs are visible.",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on the full training set and generate submission files.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    """Apply CLI arguments to the config object."""
    arg_to_config = {
        "experiment_name": "EXPERIMENT_NAME",
        "model": "MODEL_TYPE",
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "learning_rate": "LEARNING_RATE",
        "data_root": "DATA_ROOT",
        "test_data_root": "TEST_DATA_ROOT",
        "public_test_root": "PUBLIC_TEST_ROOT",
        "blind_test_root": "BLIND_TEST_ROOT",
        "val_split_file": "VAL_SPLIT_FILE",
        "seed": "SEED",
        "num_workers": "NUM_WORKERS",
        "num_frames": "NUM_FRAMES",
        "hidden_size": "HIDDEN_SIZE",
        "transformer_heads": "TRANSFORMER_HEADS",
        "transformer_layers": "TRANSFORMER_LAYERS",
        "output_dir": "OUTPUT_DIR",
    }

    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    if args.use_stn:
        config.USE_STN = True
    if args.no_stn:
        config.USE_STN = False
    if args.use_amp:
        config.USE_AMP = True
    if args.force_single_gpu:
        config.FORCE_SINGLE_GPU = True

    if args.batch_size is None:
        config.BATCH_SIZE = None
    if args.num_workers is None:
        config.NUM_WORKERS = None
    config.refresh_runtime_fields()


def resolve_data_paths(config: Config) -> None:
    """Resolve local/Kaggle train and test roots."""
    config.DATA_ROOT = str(find_train_root(config.DATA_ROOT, PROJECT_ROOT))

    generic_test = find_named_test_root("test", config.TEST_DATA_ROOT, PROJECT_ROOT)
    config.TEST_DATA_ROOT = str(generic_test) if generic_test is not None else None

    public_test = find_named_test_root(
        "Pa7a3Hin-test-public",
        config.PUBLIC_TEST_ROOT,
        PROJECT_ROOT,
    )
    config.PUBLIC_TEST_ROOT = str(public_test) if public_test is not None else None

    blind_test = find_named_test_root(
        "TKzFBtn7-test-blind",
        config.BLIND_TEST_ROOT,
        PROJECT_ROOT,
    )
    config.BLIND_TEST_ROOT = str(blind_test) if blind_test is not None else None


def make_loader(dataset: MultiFrameDataset, config: Config, shuffle: bool) -> DataLoader:
    """Create a DataLoader with consistent project defaults."""
    return DataLoader(
        dataset,
        batch_size=int(config.BATCH_SIZE or 1),
        shuffle=shuffle,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=int(config.NUM_WORKERS or 0),
        pin_memory=config.DEVICE.type == "cuda",
    )


def build_model(config: Config) -> nn.Module:
    """Build and optionally wrap the configured model."""
    if config.MODEL_TYPE == "restran":
        base_model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        )
    elif config.MODEL_TYPE == "crnn":
        base_model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}")

    base_model = base_model.to(config.DEVICE)
    if config.USE_DATA_PARALLEL:
        print(f"DataParallel enabled on {config.N_GPUS} GPUs.")
        return nn.DataParallel(base_model)
    return base_model


def load_model_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load an unwrapped checkpoint into a possibly DataParallel model."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state_dict)


def create_train_val_loaders(config: Config, submission_mode: bool) -> tuple:
    """Create train and optional validation loaders."""
    common_ds_params = {
        "split_ratio": config.SPLIT_RATIO,
        "img_height": config.IMG_HEIGHT,
        "img_width": config.IMG_WIDTH,
        "num_frames": config.NUM_FRAMES,
        "char2idx": config.CHAR2IDX,
        "val_split_file": config.VAL_SPLIT_FILE,
        "seed": config.SEED,
        "augmentation_level": config.AUGMENTATION_LEVEL,
    }

    if submission_mode:
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode="train",
            full_train=True,
            **common_ds_params,
        )
        val_loader = None
    else:
        train_ds = MultiFrameDataset(root_dir=config.DATA_ROOT, mode="train", **common_ds_params)
        val_ds = MultiFrameDataset(root_dir=config.DATA_ROOT, mode="val", **common_ds_params)
        val_loader = make_loader(val_ds, config, shuffle=False) if len(val_ds) > 0 else None
        if val_loader is None:
            print("WARNING: Validation dataset is empty.")

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Check DATA_ROOT and annotations.json files.")

    return make_loader(train_ds, config, shuffle=True), val_loader


def create_test_loader(root: str, config: Config) -> Optional[DataLoader]:
    """Create a test DataLoader for one test root."""
    if not has_tracks(root):
        return None
    test_ds = MultiFrameDataset(
        root_dir=root,
        mode="val",
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        num_frames=config.NUM_FRAMES,
        char2idx=config.CHAR2IDX,
        seed=config.SEED,
        is_test=True,
    )
    if len(test_ds) == 0:
        return None
    return make_loader(test_ds, config, shuffle=False)


def submission_roots(config: Config) -> Dict[str, str]:
    """Return deduplicated test roots in output priority order."""
    candidates = {
        "public": config.PUBLIC_TEST_ROOT,
        "blind": config.BLIND_TEST_ROOT,
        config.TEST_SET_NAME: config.TEST_DATA_ROOT,
    }
    roots: Dict[str, str] = {}
    seen = set()
    for name, root in candidates.items():
        if root is None:
            continue
        normalized = os.path.abspath(root)
        if normalized in seen:
            continue
        roots[name] = root
        seen.add(normalized)
    return roots


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    config = Config()
    apply_cli_overrides(config, args)
    resolve_data_paths(config)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.SEED)

    print("Configuration:")
    print(f"  EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"  MODEL: {config.MODEL_TYPE}")
    print(f"  USE_STN: {config.USE_STN}")
    print(f"  USE_AMP: {config.USE_AMP}")
    print(f"  DATA_ROOT: {config.DATA_ROOT}")
    print(f"  TEST_DATA_ROOT: {config.TEST_DATA_ROOT}")
    print(f"  PUBLIC_TEST_ROOT: {config.PUBLIC_TEST_ROOT}")
    print(f"  BLIND_TEST_ROOT: {config.BLIND_TEST_ROOT}")
    print(f"  EPOCHS: {config.EPOCHS}")
    print(f"  BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"  NUM_WORKERS: {config.NUM_WORKERS}")
    print(f"  DEVICE: {config.DEVICE}")
    print(f"  SUBMISSION_MODE: {args.submission_mode}")

    train_loader, val_loader = create_train_val_loaders(config, args.submission_mode)
    model = build_model(config)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model ({config.MODEL_TYPE}): {total_params:,} total params, {trainable_params:,} trainable")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )
    trainer.fit()

    if not args.submission_mode:
        return

    test_roots = submission_roots(config)
    if not test_roots:
        print("WARNING: No test data found for submission mode.")
        return

    best_model_path = os.path.join(config.OUTPUT_DIR, f"{config.EXPERIMENT_NAME}_best.pth")
    if os.path.exists(best_model_path):
        load_model_weights(model, best_model_path, config.DEVICE)
        print(f"Loaded best checkpoint: {best_model_path}")
    else:
        print(f"WARNING: Best checkpoint not found, using current model weights: {best_model_path}")

    generated: Dict[str, str] = {}
    for test_name, test_root in test_roots.items():
        test_loader = create_test_loader(test_root, config)
        if test_loader is None:
            print(f"WARNING: Test dataset '{test_name}' is empty or missing at {test_root}.")
            continue
        output_name = f"submission_{config.EXPERIMENT_NAME}_{test_name}.txt"
        generated[test_name] = trainer.predict_test(test_loader, output_filename=output_name)

    primary_name = "blind" if "blind" in generated else config.TEST_SET_NAME
    primary_path = generated.get(primary_name) or next(iter(generated.values()), None)
    if primary_path is not None:
        final_submission_path = os.path.join(config.OUTPUT_DIR, config.SUBMISSION_FILE)
        shutil.copyfile(primary_path, final_submission_path)
        print(f"Primary submission file: {final_submission_path}")


if __name__ == "__main__":
    main()
