#!/usr/bin/env python3
"""
Convert frame-wise PointPillars KITTI-format detections into the
sequence-wise CSV-style input expected by the AB3DMOT KITTI workflow.

This utility contains the project-specific conversion logic used to bridge
PointPillars outputs and AB3DMOT without redistributing AB3DMOT itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_SPLITS = {
    "val": [("0015", 376), ("0016", 209), ("0017", 145)],
    "internal_test": [("0018", 339), ("0019", 1059), ("0020", 837)],
}

CLASSES = ("Car", "Pedestrian", "Cyclist")

# AB3DMOT KITTI detection-category IDs used by the project.
CLASS_ID = {
    "Pedestrian": 1,
    "Car": 2,
    "Cyclist": 3,
}


def parse_sequence_spec(values: Iterable[str]) -> List[Tuple[str, int]]:
    """Parse entries such as 0018:339 into (sequence_id, frame_count)."""
    parsed: List[Tuple[str, int]] = []
    for value in values:
        try:
            sequence, count = value.split(":", 1)
            parsed.append((sequence, int(count)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid sequence specification '{value}'. Use SEQUENCE:FRAME_COUNT."
            ) from exc
    return parsed


def build_mapping(sequence_frames: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    mapping: List[Tuple[str, int]] = []
    for sequence, count in sequence_frames:
        mapping.extend((sequence, frame) for frame in range(count))
    return mapping


def convert(
    submit_dir: Path,
    output_dir: Path,
    sequence_frames: List[Tuple[str, int]],
    score_threshold: float,
) -> None:
    submit_files = sorted(submit_dir.glob("*.txt"))
    mapping = build_mapping(sequence_frames)

    if not submit_files:
        raise FileNotFoundError(f"No .txt detection files found in {submit_dir}")

    if len(submit_files) != len(mapping):
        raise ValueError(
            "Detection-file count does not match the configured sequence/frame mapping: "
            f"{len(submit_files)} files vs {len(mapping)} expected frames."
        )

    writers = {}
    try:
        for cls in CLASSES:
            class_dir = output_dir / f"pointpillars_{cls}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for sequence, _ in sequence_frames:
                path = class_dir / f"{sequence}.txt"
                writers[(cls, sequence)] = path.open("w", encoding="utf-8")

        for image_id, detection_file in enumerate(submit_files):
            sequence, frame = mapping[image_id]

            for line in detection_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 16:
                    continue

                cls = parts[0]
                if cls not in CLASSES:
                    continue

                alpha = float(parts[3])
                x1, y1, x2, y2 = map(float, parts[4:8])

                # PointPillars project output stores dimensions in l, h, w order.
                raw_l = float(parts[8])
                raw_h = float(parts[9])
                raw_w = float(parts[10])

                h = raw_h
                w = raw_w
                l = raw_l

                x, y, z = map(float, parts[11:14])
                rotation_y = float(parts[14])
                score = float(parts[15])

                if score < score_threshold:
                    continue

                row = [
                    frame,
                    CLASS_ID[cls],
                    x1, y1, x2, y2,
                    alpha,
                    h, w, l,
                    x, y, z,
                    rotation_y,
                    score,
                ]

                writers[(cls, sequence)].write(",".join(map(str, row)) + "\n")
    finally:
        for writer in writers.values():
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PointPillars KITTI detections to AB3DMOT sequence input."
    )
    parser.add_argument(
        "--submit_dir",
        type=Path,
        required=True,
        help="Directory containing frame-wise PointPillars KITTI-format .txt detections.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for sequence-wise AB3DMOT detection files.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(DEFAULT_SPLITS),
        default="val",
        help="Use one of the project split mappings.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        metavar="SEQUENCE:FRAME_COUNT",
        help=(
            "Optional custom sequence mapping. Repeat for each sequence, e.g. "
            "--sequence 0018:339 --sequence 0019:1059 --sequence 0020:837. "
            "When supplied, this overrides --split."
        ),
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.30,
        help="Minimum detection confidence retained for tracking input.",
    )
    args = parser.parse_args()

    sequence_frames = (
        parse_sequence_spec(args.sequence)
        if args.sequence
        else DEFAULT_SPLITS[args.split]
    )

    convert(
        submit_dir=args.submit_dir,
        output_dir=args.output_dir,
        sequence_frames=sequence_frames,
        score_threshold=args.score_threshold,
    )

    print("Conversion finished.")
    print("Sequences:", ", ".join(seq for seq, _ in sequence_frames))
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
