import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pre_process_kitti import (
    read_points,
    write_points,
    read_label,
    write_pickle,
    remove_outside_points,
    judge_difficulty,
    get_points_num_in_bbox,
    points_in_bboxes_v2,
)


VALID_CLASSES = ["Car", "Pedestrian", "Cyclist"]


def read_calib_dair(calib_path):
    """
    DAIR/KITTI-like calibration reader.

    DAIR converted files may contain only:
    P0, P1, P2, P3, R0_rect, Tr_velo_to_cam

    The original KITTI reader expects Tr_imu_to_velo also,
    which is not needed for PointPillars preprocessing here.
    """
    data = {}

    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                values = value.strip().split()
            else:
                parts = line.split()
                key = parts[0]
                values = parts[1:]

            if not values:
                continue

            data[key] = np.array([float(x) for x in values], dtype=np.float32)

    required = ["P2", "R0_rect", "Tr_velo_to_cam"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing calibration keys {missing} in {calib_path}. Available keys: {list(data.keys())}")

    calib = {}

    for key in ["P0", "P1", "P2", "P3"]:
        if key in data:
            calib[key] = data[key].reshape(3, 4)
        else:
            calib[key] = np.zeros((3, 4), dtype=np.float32)

    # Convert DAIR/KITTI-style calibration into homogeneous 4x4 matrices.
    # PointPillars geometry utilities expect square matrices for inversion.
    r0_rect_4x4 = np.eye(4, dtype=np.float32)
    r0_rect_4x4[:3, :3] = data["R0_rect"].reshape(3, 3)

    tr_velo_to_cam_4x4 = np.eye(4, dtype=np.float32)
    tr_velo_to_cam_4x4[:3, :4] = data["Tr_velo_to_cam"].reshape(3, 4)

    calib["R0_rect"] = r0_rect_4x4
    calib["Tr_velo_to_cam"] = tr_velo_to_cam_4x4

    # Dummy value for compatibility. It is not used in our preprocessing path.
    calib["Tr_imu_to_velo"] = np.eye(4, dtype=np.float32)

    return calib


def safe_read_label(label_path):
    if (not os.path.exists(label_path)) or os.path.getsize(label_path) == 0:
        return {
            "name": np.array([], dtype=object),
            "truncated": np.array([], dtype=np.float32),
            "occluded": np.array([], dtype=np.int32),
            "alpha": np.array([], dtype=np.float32),
            "bbox": np.zeros((0, 4), dtype=np.float32),
            "dimensions": np.zeros((0, 3), dtype=np.float32),
            "location": np.zeros((0, 3), dtype=np.float32),
            "rotation_y": np.array([], dtype=np.float32),
        }

    annos = read_label(label_path)

    # Keep only KITTI-compatible classes
    keep = np.array([name in VALID_CLASSES for name in annos["name"]], dtype=bool)
    for key in list(annos.keys()):
        annos[key] = annos[key][keep]

    return annos


def get_split_ids(data_root, split):
    velodyne_dir = Path(data_root) / split / "velodyne"
    ids = sorted([p.stem for p in velodyne_dir.glob("*.bin")])
    return ids


def create_data_info_pkl(data_root, split, prefix, label=True, db=False):
    sep = os.path.sep
    print(f"Processing DAIR {split} data...")

    ids = get_split_ids(data_root, split)
    print(f"{split} frames: {len(ids)}")

    infos_dict = {}

    dbinfos_train = {cls: [] for cls in VALID_CLASSES}
    if db:
        db_points_saved_path = os.path.join(data_root, f"{prefix}_gt_database")
        os.makedirs(db_points_saved_path, exist_ok=True)

    class_counter = {cls: 0 for cls in VALID_CLASSES}

    for frame_id in tqdm(ids):
        cur_info_dict = {}

        img_path = os.path.join(data_root, split, "image_2", f"{frame_id}.png")
        lidar_path = os.path.join(data_root, split, "velodyne", f"{frame_id}.bin")
        calib_path = os.path.join(data_root, split, "calib", f"{frame_id}.txt")

        cur_info_dict["velodyne_path"] = sep.join(lidar_path.split(sep)[-3:])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image_shape = img.shape[:2]
        cur_info_dict["image"] = {
            "image_shape": image_shape,
            "image_path": sep.join(img_path.split(sep)[-3:]),
            "image_idx": int(frame_id),
        }

        calib_dict = read_calib_dair(calib_path)
        cur_info_dict["calib"] = calib_dict

        lidar_points = read_points(lidar_path)
        reduced_lidar_points = remove_outside_points(
            points=lidar_points,
            r0_rect=calib_dict["R0_rect"],
            tr_velo_to_cam=calib_dict["Tr_velo_to_cam"],
            P2=calib_dict["P2"],
            image_shape=image_shape,
        )

        saved_reduced_path = os.path.join(data_root, split, "velodyne_reduced")
        os.makedirs(saved_reduced_path, exist_ok=True)
        saved_reduced_points_name = os.path.join(saved_reduced_path, f"{frame_id}.bin")
        write_points(reduced_lidar_points, saved_reduced_points_name)

        if label:
            label_path = os.path.join(data_root, split, "label_2", f"{frame_id}.txt")
            annotation_dict = safe_read_label(label_path)

            n_valid_bbox = len(annotation_dict["name"])

            if n_valid_bbox > 0:
                annotation_dict["difficulty"] = judge_difficulty(annotation_dict)
                annotation_dict["num_points_in_gt"] = get_points_num_in_bbox(
                    points=reduced_lidar_points,
                    r0_rect=calib_dict["R0_rect"],
                    tr_velo_to_cam=calib_dict["Tr_velo_to_cam"],
                    dimensions=annotation_dict["dimensions"],
                    location=annotation_dict["location"],
                    rotation_y=annotation_dict["rotation_y"],
                    name=annotation_dict["name"],
                )
            else:
                annotation_dict["difficulty"] = np.array([], dtype=np.int32)
                annotation_dict["num_points_in_gt"] = np.array([], dtype=np.int32)

            cur_info_dict["annos"] = annotation_dict

            if db and n_valid_bbox > 0:
                indices, n_total_bbox, n_valid_bbox, boxes_lidar, name = points_in_bboxes_v2(
                    points=lidar_points,
                    r0_rect=calib_dict["R0_rect"].astype(np.float32),
                    tr_velo_to_cam=calib_dict["Tr_velo_to_cam"].astype(np.float32),
                    dimensions=annotation_dict["dimensions"].astype(np.float32),
                    location=annotation_dict["location"].astype(np.float32),
                    rotation_y=annotation_dict["rotation_y"].astype(np.float32),
                    name=annotation_dict["name"],
                )

                for j in range(n_valid_bbox):
                    cls_name = name[j]
                    if cls_name not in VALID_CLASSES:
                        continue

                    class_counter[cls_name] += 1

                    db_points = lidar_points[indices[:, j]]
                    db_points[:, :3] -= boxes_lidar[j, :3]

                    db_points_saved_name = os.path.join(
                        db_points_saved_path,
                        f"{int(frame_id)}_{cls_name}_{j}.bin",
                    )
                    write_points(db_points, db_points_saved_name)

                    db_info = {
                        "name": cls_name,
                        "path": os.path.join(os.path.basename(db_points_saved_path), f"{int(frame_id)}_{cls_name}_{j}.bin"),
                        "box3d_lidar": boxes_lidar[j],
                        "difficulty": annotation_dict["difficulty"][j],
                        "num_points_in_gt": len(db_points),
                    }
                    dbinfos_train[cls_name].append(db_info)

        infos_dict[int(frame_id)] = cur_info_dict

    saved_path = os.path.join(data_root, f"{prefix}_infos_{split}.pkl")
    write_pickle(infos_dict, saved_path)

    if db:
        saved_db_path = os.path.join(data_root, f"{prefix}_dbinfos_train.pkl")
        write_pickle(dbinfos_train, saved_db_path)

        print("DAIR GT database class counts:")
        for cls in VALID_CLASSES:
            print(f"{cls}: {len(dbinfos_train[cls])}")

    return infos_dict


def main(args):
    data_root = args.data_root
    prefix = args.prefix

    train_infos = create_data_info_pkl(data_root, "train", prefix, label=True, db=True)
    val_infos = create_data_info_pkl(data_root, "val", prefix, label=True, db=False)

    trainval_infos = {**train_infos, **val_infos}
    write_pickle(trainval_infos, os.path.join(data_root, f"{prefix}_infos_trainval.pkl"))

    create_data_info_pkl(data_root, "test", prefix, label=False, db=False)

    print("DAIR PointPillars preprocessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAIR KITTI-like preprocessing for PointPillars")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--prefix", default="kitti")
    args = parser.parse_args()
    main(args)
