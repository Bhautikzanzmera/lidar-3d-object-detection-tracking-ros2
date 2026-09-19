import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from pre_process_kitti import (
    read_points,
    write_points,
    read_calib,
    read_label,
    write_pickle,
    remove_outside_points,
    judge_difficulty,
    get_points_num_in_bbox,
    points_in_bboxes_v2,
)


def read_split_ids(data_root, data_type):
    ids_file = os.path.join(data_root, "ImageSets", f"{data_type}.txt")
    with open(ids_file, "r") as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]
    return ids


def create_data_info_pkl(data_root, data_type, prefix, label=True, db=False):
    sep = os.path.sep
    print(f"Processing project KITTI {data_type} data...")

    ids = read_split_ids(data_root, data_type)
    print(f"{data_type} frames: {len(ids)}")

    # Professor train/val/test all come from KITTI labeled training folder
    source_split = "training"

    kitti_infos_dict = {}

    if db:
        kitti_dbinfos_train = {}
        db_points_saved_path = os.path.join(data_root, f"{prefix}_gt_database")
        os.makedirs(db_points_saved_path, exist_ok=True)

    for frame_id in tqdm(ids):
        cur_info_dict = {}

        img_path = os.path.join(data_root, source_split, "image_2", f"{frame_id}.png")
        lidar_path = os.path.join(data_root, source_split, "velodyne", f"{frame_id}.bin")
        calib_path = os.path.join(data_root, source_split, "calib", f"{frame_id}.txt")

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

        calib_dict = read_calib(calib_path)
        cur_info_dict["calib"] = calib_dict

        lidar_points = read_points(lidar_path)
        reduced_lidar_points = remove_outside_points(
            points=lidar_points,
            r0_rect=calib_dict["R0_rect"],
            tr_velo_to_cam=calib_dict["Tr_velo_to_cam"],
            P2=calib_dict["P2"],
            image_shape=image_shape,
        )

        saved_reduced_path = os.path.join(data_root, source_split, "velodyne_reduced")
        os.makedirs(saved_reduced_path, exist_ok=True)
        saved_reduced_points_name = os.path.join(saved_reduced_path, f"{frame_id}.bin")
        write_points(reduced_lidar_points, saved_reduced_points_name)

        if label:
            label_path = os.path.join(data_root, source_split, "label_2", f"{frame_id}.txt")
            annotation_dict = read_label(label_path)
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
            cur_info_dict["annos"] = annotation_dict

            if db:
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
                    db_points = lidar_points[indices[:, j]]
                    db_points[:, :3] -= boxes_lidar[j, :3]
                    db_points_saved_name = os.path.join(
                        db_points_saved_path,
                        f"{int(frame_id)}_{name[j]}_{j}.bin",
                    )
                    write_points(db_points, db_points_saved_name)

                    db_info = {
                        "name": name[j],
                        "path": os.path.join(
                            os.path.basename(db_points_saved_path),
                            f"{int(frame_id)}_{name[j]}_{j}.bin",
                        ),
                        "box3d_lidar": boxes_lidar[j],
                        "difficulty": annotation_dict["difficulty"][j],
                        "num_points_in_gt": len(db_points),
                    }

                    if name[j] not in kitti_dbinfos_train:
                        kitti_dbinfos_train[name[j]] = [db_info]
                    else:
                        kitti_dbinfos_train[name[j]].append(db_info)

        kitti_infos_dict[int(frame_id)] = cur_info_dict

    saved_path = os.path.join(data_root, f"{prefix}_infos_{data_type}.pkl")
    write_pickle(kitti_infos_dict, saved_path)

    if db:
        saved_db_path = os.path.join(data_root, f"{prefix}_dbinfos_train.pkl")
        write_pickle(kitti_dbinfos_train, saved_db_path)

    return kitti_infos_dict


def main(args):
    data_root = args.data_root
    prefix = args.prefix

    train_infos = create_data_info_pkl(data_root, "train", prefix, label=True, db=True)
    val_infos = create_data_info_pkl(data_root, "val", prefix, label=True, db=False)

    trainval_infos = {**train_infos, **val_infos}
    write_pickle(trainval_infos, os.path.join(data_root, f"{prefix}_infos_trainval.pkl"))

    # Professor test split is labeled, so we also include labels for later test evaluation.
    create_data_info_pkl(data_root, "test", prefix, label=True, db=False)

    print("KITTI project split preprocessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI project 8:1:1 preprocessing for PointPillars")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--prefix", default="kitti")
    args = parser.parse_args()
    main(args)
