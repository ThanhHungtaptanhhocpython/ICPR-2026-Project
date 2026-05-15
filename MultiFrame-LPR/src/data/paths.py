"""Dataset path discovery for local and Kaggle layouts."""
import glob
from pathlib import Path
from typing import Iterable, List, Optional


def has_tracks(root: Optional[str | Path]) -> bool:
    """Return True when root contains track_* folders recursively."""
    if root is None:
        return False
    root_path = Path(root)
    if not root_path.exists():
        return False
    return bool(glob.glob(str(root_path / "**" / "track_*"), recursive=True))


def _dedupe(paths: Iterable[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def existing_dataset_roots(project_root: Optional[str | Path] = None) -> List[Path]:
    """Return candidate dataset roots used by this repo and Kaggle."""
    project_root = Path(project_root or ".").resolve()
    kaggle_input = Path("/kaggle/input")

    roots: List[Path] = [
        project_root / "data" / "LRLPR-26-5opEvJTW (1)",
        project_root / "data" / "LRLPR-26-5opEvJTW",
        project_root / "data",
        Path("/kaggle/input/datasets/trunghiu/icpr-car-plate-dataset"),
        Path("/kaggle/input/icpr-car-plate-dataset"),
        Path("/kaggle/input/ICPR_CAR_PLATE_DATASET"),
    ]

    if kaggle_input.exists():
        roots.extend(path for path in sorted(kaggle_input.glob("*")) if path.is_dir())
        trunghiu_root = kaggle_input / "datasets" / "trunghiu"
        if trunghiu_root.exists():
            roots.extend(path for path in sorted(trunghiu_root.glob("*")) if path.is_dir())

    return _dedupe(roots)


def find_train_root(
    explicit: Optional[str | Path] = None,
    project_root: Optional[str | Path] = None,
) -> Path:
    """Find a train root from an explicit path, local repo data, or Kaggle input."""
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))

    for dataset_root in existing_dataset_roots(project_root):
        candidates.extend([
            dataset_root / "data" / "train",
            dataset_root / "train",
        ])

    for candidate in _dedupe(candidates):
        if has_tracks(candidate):
            return candidate

    raise FileNotFoundError(
        "Cannot find training data. Expected a train folder containing track_* directories."
    )


def find_named_test_root(
    name: str,
    explicit: Optional[str | Path] = None,
    project_root: Optional[str | Path] = None,
) -> Optional[Path]:
    """Find a named test root such as test, Pa7a3Hin-test-public, or TKzFBtn7-test-blind."""
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))

    for dataset_root in existing_dataset_roots(project_root):
        candidates.extend([
            dataset_root / name / name,
            dataset_root / name,
            dataset_root / "data" / name,
        ])

    for candidate in _dedupe(candidates):
        if has_tracks(candidate):
            return candidate
    return None
