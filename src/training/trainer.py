"""Trainer class encapsulating training, validation, and inference."""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class Trainer:
    """Encapsulates OCR training, validation, checkpointing, and inference."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        self.use_amp = self.device.type == "cuda" and bool(getattr(config, "USE_AMP", False))
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=max(1, len(train_loader)),
            epochs=config.EPOCHS,
        )
        try:
            self.scaler = GradScaler("cuda", enabled=self.use_amp)
        except TypeError:
            self.scaler = GradScaler(enabled=self.use_amp)

        self.best_acc = 0.0
        self.current_epoch = 0

    def _output_path(self, filename: str) -> str:
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        return os.path.join(self.config.OUTPUT_DIR, filename)

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.EPOCHS}")

        for images, targets, target_lengths, _, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(self.autocast_device_type, enabled=self.use_amp):
                preds = self.model(images)
                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long,
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= old_scale:
                self.scheduler.step()

            epoch_loss += float(loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

        return epoch_loss / max(1, len(self.train_loader))

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Run validation and return metrics plus submission-style rows."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"loss": 0.0, "acc": 0.0}, []

        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []

        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids in tqdm(
                self.val_loader,
                desc="Validation",
            ):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                preds = self.model(images)

                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long,
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths,
                )
                val_loss += float(loss.item())

                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_ids[i]},{pred_text};{conf:.4f}")
                total_samples += len(labels_text)

        return {
            "loss": val_loss / max(1, len(self.val_loader)),
            "acc": 100.0 * total_correct / max(1, total_samples),
        }, submission_data

    def save_submission(self, submission_data: List[str], filename: Optional[str] = None) -> str:
        """Save submission rows and return the output path."""
        filename = filename or f"submission_{self.config.EXPERIMENT_NAME}_val.txt"
        path = self._output_path(filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(submission_data))
        print(f"Saved {len(submission_data)} lines to {path}")
        return path

    def save_model(self, filename: Optional[str] = None) -> str:
        """Save an unwrapped model state dict and return the checkpoint path."""
        filename = filename or f"{self.config.EXPERIMENT_NAME}_best.pth"
        path = self._output_path(filename)
        torch.save(self._unwrap_model().state_dict(), path)
        print(f"Saved model: {path}")
        return path

    def fit(self) -> None:
        """Run the full training loop."""
        print(f"Training on {self.device} for {self.config.EPOCHS} epochs")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            train_loss = self.train_one_epoch()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            val_metrics, submission_data = self.validate()
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['acc']:.2f}%"
            )

            if self.val_loader is not None and val_metrics["acc"] > self.best_acc:
                self.best_acc = val_metrics["acc"]
                self.save_model()
                if submission_data:
                    self.save_submission(submission_data)

        if self.val_loader is None:
            self.save_model()

        print(f"Training complete. Best validation accuracy: {self.best_acc:.2f}%")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on a data loader."""
        self.model.eval()
        results: List[Tuple[str, str, float]] = []

        with torch.no_grad():
            for images, _, _, _, track_ids in loader:
                images = images.to(self.device, non_blocking=True)
                preds = self.model(images)

                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))

        return results

    def predict_test(self, test_loader: DataLoader, output_filename: Optional[str] = None) -> str:
        """Run inference on test data and save a submission file."""
        self.model.eval()
        rows: List[str] = []

        with torch.no_grad():
            for images, _, _, _, track_ids in tqdm(test_loader, desc="Test inference"):
                images = images.to(self.device, non_blocking=True)
                preds = self.model(images)
                decoded_list = decode_with_confidence(preds, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    rows.append(f"{track_ids[i]},{pred_text};{conf:.4f}")

        output_filename = output_filename or f"submission_{self.config.EXPERIMENT_NAME}_test.txt"
        return self.save_submission(rows, output_filename)
