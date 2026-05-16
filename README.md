# MultiFrame-LPR

Multi-frame OCR pipeline for the ICPR 2026 Low-Resolution License Plate Recognition challenge.

## Competition

[ICPR 2026 Low-Resolution License Plate Recognition](https://www.codabench.org/competitions/12259/#/results-tab) | Team: `AIO_AIZO` (Soloist) | Public Score: `0.745` (Rank 22) | Blind Score: `0.75` (Rank 32) | Track: Computer Vision

- **Multi-frame OCR**: Built a Kaggle-ready **ResTranOCR** pipeline for low-resolution license plate recognition.
- **Competition result**: Achieved competitive leaderboard results on both public and blind test sets.

The repo integrates the Kaggle notebook workflow with reusable code modules for dataset discovery, training, validation, and submission generation. The default local dataset path is:

```text
data/LRLPR-26-5opEvJTW (1)/
|-- train/
|   |-- Scenario-A/
|   `-- Scenario-B/
`-- test/
    `-- track_*/
```

Kaggle dataset mounts such as `/kaggle/input/datasets/trunghiu/icpr-car-plate-dataset` are auto-detected when the local path is not available.

## Quick Start

```bash
uv sync
python train.py
```

Train a CRNN baseline:

```bash
python train.py --model crnn --experiment-name crnn_baseline
```

Train on all labeled data and generate submission files:

```bash
python train.py --submission-mode
```

Enable STN only when you want to test it explicitly:

```bash
python train.py --use-stn
```

## Main Options

- `--data-root`: training root containing `track_*` folders recursively.
- `--test-data-root`: generic test root for submission mode.
- `--public-test-root`: Kaggle public test root.
- `--blind-test-root`: Kaggle blind test root.
- `--num-frames`: number of frames loaded per track, default `5`.
- `--aug-level`: `full` or `light`.
- `--use-stn` / `--no-stn`: toggle Spatial Transformer Network alignment.
- `--use-amp`: enable CUDA mixed precision.
- `--force-single-gpu`: disable `DataParallel` on multi-GPU machines.
- `--output-dir`: directory for checkpoints and submissions, default `results/`.

## Pipeline

- `src/data/paths.py`: local/Kaggle dataset discovery.
- `src/data/dataset.py`: recursive multi-frame dataset with Scenario-B validation split, frame selection/padding, label cleaning, and test indexing.
- `src/data/transforms.py`: Albumentations transforms applied consistently across all frames in a track.
- `src/models/`: CRNN and ResTranOCR architectures.
- `src/training/trainer.py`: training, validation, checkpointing, DataParallel-safe saving, and test inference.
- `train.py`: CLI entry point.
- `run_ablation.py`: CRNN/ResTranOCR experiments with and without STN.

## Outputs

Training writes to `results/` by default:

- `{experiment_name}_best.pth`
- `submission_{experiment_name}_val.txt`
- `submission_{experiment_name}_{test|public|blind}.txt`
- `submission.txt` in submission mode, preferring blind test predictions when available.
