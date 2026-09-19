# PointPillars Project Integration

This directory contains project-specific preprocessing, evaluation, and sanity-check scripts used with the PointPillars implementation during the LiDAR 3D detection and tracking project.

## Upstream dependency

The detector implementation used by this project is based on the PyTorch PointPillars repository by `zhulf0804`:

https://github.com/zhulf0804/PointPillars

The full upstream detector source is not duplicated here. Obtain it from the upstream repository and follow its installation instructions. The upstream MIT license is preserved in `POINTPILLARS_UPSTREAM_LICENSE`.

## Included scripts

- `pre_process_kitti_project_split.py` — prepares the project-specific KITTI train/validation/internal-test split
- `pre_process_dair_pointpillars.py` — converts the prepared DAIR-V2X-I KITTI-like data into PointPillars information files
- `evaluate_project_split.py` — evaluates a PointPillars checkpoint on the project validation or internal-test split
- `train_sanity_check.py` — runs a single-batch forward/backward pass to verify that the training pipeline is functioning

## Example usage

```bash
python pre_process_kitti_project_split.py --data_root /path/to/kitti
python train_sanity_check.py --data_root /path/to/kitti
python evaluate_project_split.py \
  --data_root /path/to/kitti \
  --ckpt /path/to/checkpoint.pth \
  --split val \
  --saved_path results
```

For DAIR-V2X-I:

```bash
python pre_process_dair_pointpillars.py --data_root /path/to/dair_kitti_like
```

## Notes

Dataset files and model checkpoints are intentionally excluded from this repository. Paths are supplied at runtime so the code does not depend on the original development machine.

See `environment/pointpillars/requirements.txt` in the repository root for the legacy dependency versions used by the project.
