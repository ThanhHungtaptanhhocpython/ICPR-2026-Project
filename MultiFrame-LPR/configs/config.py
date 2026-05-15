"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass
class Config:
    """Training configuration with all hyperparameters."""
    
    # Experiment tracking
    MODEL_TYPE: str = "restran"  # "crnn" or "restran"
    EXPERIMENT_NAME: str = "restran_t4_kaggle"
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    # Kaggle T4/DataParallel can be unstable with STN grid_sample. Enable after a stable baseline run.
    USE_STN: bool = False
    
    # Data paths
    DATA_ROOT: str = "data/LRLPR-26-5opEvJTW (1)/train"
    TEST_DATA_ROOT: Optional[str] = "data/LRLPR-26-5opEvJTW (1)/test"
    PUBLIC_TEST_ROOT: Optional[str] = None
    BLIND_TEST_ROOT: Optional[str] = None
    TEST_SET_NAME: str = "test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"
    
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    NUM_FRAMES: int = 5
    
    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Training hyperparameters
    BATCH_SIZE_SINGLE_GPU: int = 32
    BATCH_SIZE_DUAL_GPU: int = 64
    BATCH_SIZE_CPU: int = 8
    BATCH_SIZE: Optional[int] = None
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30
    SEED: int = 42
    NUM_WORKERS_SINGLE_GPU: int = 2
    NUM_WORKERS_DUAL_GPU: int = 4
    NUM_WORKERS: Optional[int] = None
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False
    USE_AMP: bool = False
    
    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25
    
    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1
    
    FORCE_SINGLE_GPU: bool = False
    DEVICE: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    N_GPUS: int = field(default_factory=lambda: torch.cuda.device_count() if torch.cuda.is_available() else 0)
    USE_DATA_PARALLEL: bool = field(default=False, init=False)
    OUTPUT_DIR: str = "results"
    
    # Derived attributes (computed in __post_init__)
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank
        self.refresh_runtime_fields()

    def refresh_runtime_fields(self) -> None:
        """Refresh hardware-dependent fields after CLI overrides."""
        self.USE_DATA_PARALLEL = (
            self.DEVICE.type == "cuda"
            and self.N_GPUS > 1
            and not self.FORCE_SINGLE_GPU
        )
        if self.BATCH_SIZE is None:
            if self.USE_DATA_PARALLEL:
                self.BATCH_SIZE = self.BATCH_SIZE_DUAL_GPU
            elif self.DEVICE.type == "cuda":
                self.BATCH_SIZE = self.BATCH_SIZE_SINGLE_GPU
            else:
                self.BATCH_SIZE = self.BATCH_SIZE_CPU
        if self.NUM_WORKERS is None:
            self.NUM_WORKERS = (
                self.NUM_WORKERS_DUAL_GPU
                if self.USE_DATA_PARALLEL
                else self.NUM_WORKERS_SINGLE_GPU
            )


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()
