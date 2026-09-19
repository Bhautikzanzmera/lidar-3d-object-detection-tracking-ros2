# AB3DMOT Tracking Integration

This directory contains the project-specific bridge between PointPillars detection output and the AB3DMOT multi-object tracking workflow.

## Why AB3DMOT source is not included

AB3DMOT is an external research dependency and is **not redistributed** in this portfolio repository. Obtain it from the upstream project:

https://github.com/xinshuoweng/AB3DMOT

Use AB3DMOT according to its upstream license and installation instructions.

## Included utility

`convert_pointpillars_to_ab3dmot.py` converts frame-wise PointPillars KITTI-format detections into sequence-wise detection files used by the KITTI AB3DMOT workflow.

The conversion preserves:

- frame index
- object category ID
- 2D bounding box
- observation angle (`alpha`)
- 3D dimensions
- 3D location
- yaw / rotation
- detector confidence score

The project also corrected the detector/tracker dimension-order mismatch so that `(h, w, l)` is written in the order expected by the tracking input.

## Project sequence splits

Validation:

- `0015` — 376 frames
- `0016` — 209 frames
- `0017` — 145 frames

Internal test / visualization:

- `0018` — 339 frames
- `0019` — 1059 frames
- `0020` — 837 frames

## Example

Validation split:

```bash
python convert_pointpillars_to_ab3dmot.py \
  --submit_dir /path/to/pointpillars/submit \
  --output_dir /path/to/AB3DMOT/data/KITTI/detection \
  --split val
```

Internal test split:

```bash
python convert_pointpillars_to_ab3dmot.py \
  --submit_dir /path/to/pointpillars/submit \
  --output_dir /path/to/AB3DMOT/data/KITTI/detection \
  --split internal_test
```

Custom sequence mapping is also supported:

```bash
python convert_pointpillars_to_ab3dmot.py \
  --submit_dir /path/to/submit \
  --output_dir /path/to/output \
  --sequence 0018:339 \
  --sequence 0019:1059 \
  --sequence 0020:837
```

## Tracking stage

After conversion, place the generated detection folders where your AB3DMOT KITTI configuration expects them, then run the upstream tracker for the corresponding split and detection name.

The portfolio repository intentionally keeps the conversion/integration logic separate from the third-party tracker source.
