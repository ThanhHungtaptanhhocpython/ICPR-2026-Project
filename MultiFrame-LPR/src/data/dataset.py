"""Dataset for multi-frame license plate recognition."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_degradation_transforms,
    get_light_transforms,
    get_train_transforms,
    get_val_transforms,
)


class MultiFrameDataset(Dataset):
    """Load track folders as fixed-length multi-frame OCR samples."""

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        num_frames: int = 5,
        char2idx: Optional[Dict[str, int]] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing track folders, recursively.
            mode: "train" or "val".
            split_ratio: Train/validation split ratio.
            img_height: Target image height.
            img_width: Target image width.
            num_frames: Number of frames returned per track.
            char2idx: Character to index mapping.
            val_split_file: JSON file storing validation track keys.
            seed: Random seed for reproducible splitting.
            augmentation_level: "full" or "light" augmentation for training.
            is_test: If True, load test data without labels.
            full_train: If True, use all labeled tracks for training.
        """
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.num_frames = num_frames
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train

        if mode == "train":
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width, num_frames)
            else:
                self.transform = get_train_transforms(img_height, img_width, num_frames)
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms(img_height, img_width, num_frames)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {self.root_dir}")
        all_tracks = sorted(glob.glob(os.path.join(self.root_dir, "**", "track_*"), recursive=True))
        if not all_tracks:
            print("ERROR: No track_* folders found.")
            return

        if is_test:
            self._index_test_samples(all_tracks)
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            selected_tracks = train_tracks if mode == "train" else val_tracks
            self._index_labeled_samples(selected_tracks)

        print(f"[{mode.upper()}] Total samples: {len(self.samples)}")

    def _track_key(self, track_path: str) -> str:
        """Return a stable split key relative to the dataset root."""
        return os.path.relpath(track_path, self.root_dir).replace(os.sep, "/")

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float,
    ) -> Tuple[List[str], List[str]]:
        """Load an existing split or create one, prioritizing Scenario-B for validation."""
        if self.full_train:
            print("FULL TRAIN MODE: Using all tracks for training.")
            return all_tracks, []

        train_tracks: List[str] = []
        val_tracks: List[str] = []

        if os.path.exists(self.val_split_file):
            print(f"Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, "r", encoding="utf-8") as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for track_path in all_tracks:
                key = self._track_key(track_path)
                if key in val_ids or os.path.basename(track_path) in val_ids:
                    val_tracks.append(track_path)
                else:
                    train_tracks.append(track_path)

        if not val_tracks:
            print("Creating validation split from Scenario-B when available.")
            scenario_b_tracks = [
                track_path
                for track_path in all_tracks
                if "Scenario-B" in track_path.replace("\\", "/")
            ]
            split_pool = list(scenario_b_tracks or all_tracks)
            val_size = max(1, int(len(split_pool) * (1 - split_ratio)))

            random.Random(self.seed).shuffle(split_pool)
            val_tracks = split_pool[:val_size]
            val_set = set(val_tracks)
            train_tracks = [track_path for track_path in all_tracks if track_path not in val_set]

            split_dir = os.path.dirname(self.val_split_file)
            if split_dir:
                os.makedirs(split_dir, exist_ok=True)
            with open(self.val_split_file, "w", encoding="utf-8") as f:
                json.dump([self._track_key(track_path) for track_path in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _select_frames(self, paths: List[str]) -> List[str]:
        """Select or pad frame paths to exactly num_frames entries."""
        paths = sorted(paths)
        if not paths:
            return []
        if len(paths) >= self.num_frames:
            indices = np.linspace(0, len(paths) - 1, self.num_frames).round().astype(int)
            return [paths[index] for index in indices]
        return paths + [paths[-1]] * (self.num_frames - len(paths))

    def _frame_paths(self, track_path: str, prefix: str) -> List[str]:
        paths = (
            glob.glob(os.path.join(track_path, f"{prefix}-*.png"))
            + glob.glob(os.path.join(track_path, f"{prefix}-*.jpg"))
        )
        return self._select_frames(paths)

    def _read_label(self, track_path: str) -> Optional[str]:
        json_path = os.path.join(track_path, "annotations.json")
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = data[0]
            label = str(data.get("plate_text", data.get("license_plate", data.get("text", ""))))
            label = label.strip().upper()
            label = "".join(char for char in label if char in self.char2idx)
            return label or None
        except Exception:
            return None

    def _index_labeled_samples(self, tracks: List[str]) -> None:
        """Index labeled LR and optional synthetic-HR samples."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            label = self._read_label(track_path)
            if not label:
                continue

            track_id = os.path.basename(track_path)
            lr_files = self._frame_paths(track_path, "lr")
            hr_files = self._frame_paths(track_path, "hr")

            if lr_files:
                self.samples.append({
                    "paths": lr_files,
                    "label": label,
                    "is_synthetic": False,
                    "track_id": track_id,
                })

            if self.mode == "train" and hr_files:
                self.samples.append({
                    "paths": hr_files,
                    "label": label,
                    "is_synthetic": True,
                    "track_id": track_id,
                })

    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test samples without using labels."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            lr_files = self._frame_paths(track_path, "lr")
            if lr_files:
                self.samples.append({
                    "paths": lr_files,
                    "label": "",
                    "is_synthetic": False,
                    "track_id": os.path.basename(track_path),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _align_frame_shapes(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Align raw frame sizes before Albumentations validates additional targets."""
        if not images:
            return images
        base_h, base_w = images[0].shape[:2]
        aligned = []
        for image in images:
            h, w = image.shape[:2]
            if h != base_h or w != base_w:
                image = cv2.resize(image, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            aligned.append(image)
        return aligned

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """Load one track as a tensor of shape [num_frames, 3, height, width]."""
        item = self.samples[idx]

        images = []
        for img_path in item["paths"]:
            image = self._load_rgb(img_path)
            if item["is_synthetic"] and self.degrade is not None:
                image = self.degrade(image=image)["image"]
            images.append(image)
        images = self._align_frame_shapes(images)

        transform_kwargs = {"image": images[0]}
        for frame_idx in range(1, self.num_frames):
            transform_kwargs[f"image{frame_idx}"] = images[frame_idx]
        augmented = self.transform(**transform_kwargs)
        frames = [augmented["image"]]
        frames.extend(augmented[f"image{frame_idx}"] for frame_idx in range(1, self.num_frames))
        images_tensor = torch.stack(frames, dim=0)

        label = item["label"]
        if self.is_test:
            target = [0]
            target_len = 1
        else:
            target = [self.char2idx[char] for char in label if char in self.char2idx]
            target = target or [0]
            target_len = len(target)

        return images_tensor, torch.tensor(target, dtype=torch.long), target_len, label, item["track_id"]

    @staticmethod
    def collate_fn(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...]]:
        """Custom collate function for variable-length CTC targets."""
        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        return (
            torch.stack(images, 0),
            torch.cat(targets),
            torch.tensor(target_lengths, dtype=torch.long),
            labels_text,
            track_ids,
        )
