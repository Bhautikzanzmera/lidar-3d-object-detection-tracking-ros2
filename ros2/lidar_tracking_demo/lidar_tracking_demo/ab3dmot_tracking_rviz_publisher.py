#!/usr/bin/env python3

import os
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class AB3DMOTTrackingRvizPublisher(Node):
    def __init__(self):
        super().__init__("ab3dmot_tracking_rviz_publisher")

        self.declare_parameter("kitti_tracking_root", "")
        self.declare_parameter("ab3dmot_result_root", "")
        self.declare_parameter("calib_file", "")
        self.declare_parameter("sequence", "0019")
        self.declare_parameter("frame_id", "velodyne")
        self.declare_parameter("fps", 0.5)
        self.declare_parameter("loop", True)
        self.declare_parameter("max_points", 0)

        # Clean recording filter
        self.declare_parameter("demo_filter", True)
        self.declare_parameter("max_demo_tracks", 6)
        self.declare_parameter("min_score", 0.50)
        self.declare_parameter("max_forward_m", 32.0)
        self.declare_parameter("max_side_m", 11.0)

        self.kitti_tracking_root = self.get_parameter("kitti_tracking_root").value
        self.ab3dmot_result_root = self.get_parameter("ab3dmot_result_root").value
        self.calib_file_override = self.get_parameter("calib_file").value
        self.sequence = str(self.get_parameter("sequence").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.max_points = int(self.get_parameter("max_points").value)

        self.demo_filter = bool(self.get_parameter("demo_filter").value)
        self.max_demo_tracks = int(self.get_parameter("max_demo_tracks").value)
        self.min_score = float(self.get_parameter("min_score").value)
        self.max_forward_m = float(self.get_parameter("max_forward_m").value)
        self.max_side_m = float(self.get_parameter("max_side_m").value)

        if not self.kitti_tracking_root:
            raise ValueError('Set kitti_tracking_root to your KITTI tracking training folder.')
        if not self.ab3dmot_result_root:
            raise ValueError('Set ab3dmot_result_root to the folder containing sequence tracking result files.')

        self.points_pub = self.create_publisher(PointCloud2, "/points_raw", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/ab3dmot_tracking_boxes", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.overlay_pub = self.create_publisher(Image, "/camera/tracking_overlay", 10)

        self.velodyne_dir = os.path.join(self.kitti_tracking_root, "velodyne", self.sequence)
        self.image_dir = os.path.join(self.kitti_tracking_root, "image_02", self.sequence)
        raw_calib_file = os.path.join(self.kitti_tracking_root, "calib", f"{self.sequence}.txt")

        # By default use the calibration file from the selected KITTI tracking root.
        # If your detection/tracking conversion used a different calibration file, pass
        # it explicitly with the ROS 2 parameter: calib_file:=/path/to/sequence.txt
        if self.calib_file_override:
            self.calib_file = os.path.expanduser(self.calib_file_override)
        else:
            self.calib_file = raw_calib_file
        self.result_file = os.path.join(self.ab3dmot_result_root, f"{self.sequence}.txt")

        self.frames = sorted([
            int(os.path.splitext(f)[0])
            for f in os.listdir(self.velodyne_dir)
            if f.endswith(".bin")
        ])

        self.cam_to_velo = self.load_cam_to_velo(self.calib_file)
        self.tracks_by_frame = self.load_tracks(self.result_file)

        self.frame_index = 0

        self.get_logger().info(f"Sequence: {self.sequence}")
        self.get_logger().info(f"LiDAR frames: {len(self.frames)}")
        self.get_logger().info(f"AB3DMOT result file: {self.result_file}")
        self.get_logger().info(f"Calibration file: {self.calib_file}")
        self.get_logger().info(f"Tracking boxes loaded: {sum(len(v) for v in self.tracks_by_frame.values())}")
        self.get_logger().info("RViz topics: /points_raw, /ab3dmot_tracking_boxes, /camera/image_raw, /camera/tracking_overlay")

        period = 1.0 / self.fps if self.fps > 0 else 1.0
        self.timer = self.create_timer(period, self.publish_frame)

    def class_color(self, cls):
        # Same colors for LiDAR view and image view
        if cls == "Car":
            return 0.0, 1.0, 0.0          # green
        if cls == "Pedestrian":
            return 1.0, 0.35, 0.0         # orange
        if cls == "Cyclist":
            return 0.0, 0.35, 1.0         # blue
        return 1.0, 1.0, 1.0

    def load_cam_to_velo(self, calib_file):
        data = {}
        with open(calib_file, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                vals = np.array([float(x) for x in value.strip().split()], dtype=np.float64)
                data[key.strip()] = vals

        if "R0_rect" in data:
            R = data["R0_rect"].reshape(3, 3)
        elif "R_rect" in data:
            R = data["R_rect"].reshape(3, 3)
        else:
            R = np.eye(3)

        # KITTI tracking calibration files may use slightly different names.
        Tr = None
        for key in ["Tr_velo_to_cam", "Tr_velo_cam", "Tr_velo2cam", "Tr_velo"]:
            if key in data and data[key].size == 12:
                Tr = data[key].reshape(3, 4)
                break

        # Fallback: search any key containing both "velo" and "cam"
        if Tr is None:
            for key, vals in data.items():
                if "velo" in key.lower() and "cam" in key.lower() and vals.size == 12:
                    Tr = vals.reshape(3, 4)
                    break

        if Tr is None:
            raise RuntimeError(
                f"Could not find velo-to-camera transform in calibration file: {calib_file}. "
                f"Available keys: {list(data.keys())}"
            )

        R4 = np.eye(4)
        R4[:3, :3] = R

        Tr4 = np.eye(4)
        Tr4[:3, :4] = Tr

        velo_to_cam = R4 @ Tr4
        cam_to_velo = np.linalg.inv(velo_to_cam)
        return cam_to_velo

    def load_tracks(self, result_file):
        tracks = {}
        if not os.path.exists(result_file):
            self.get_logger().warn(f"Result file not found: {result_file}")
            return tracks

        with open(result_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17:
                    continue

                try:
                    frame = int(float(parts[0]))
                    track_id = str(parts[1])
                    cls = parts[2]

                    x1 = float(parts[6])
                    y1 = float(parts[7])
                    x2 = float(parts[8])
                    y2 = float(parts[9])

                    h = float(parts[10])
                    w = float(parts[11])
                    l = float(parts[12])

                    x = float(parts[13])
                    y = float(parts[14])
                    z = float(parts[15])
                    ry = float(parts[16])

                    # AB3DMOT final output may not store detection confidence as the last column.
                    # For visualization, do not reject tracks because of a wrongly parsed score.
                    score = 1.0
                    if len(parts) >= 18:
                        try:
                            possible_score = float(parts[17])
                            if 0.0 <= possible_score <= 1.0:
                                score = possible_score
                        except Exception:
                            score = 1.0
                except Exception:
                    continue

                trk = {
                    "frame": frame,
                    "track_id": track_id,
                    "cls": cls,
                    "bbox": [x1, y1, x2, y2],
                    "h": h,
                    "w": w,
                    "l": l,
                    "x": x,
                    "y": y,
                    "z": z,
                    "ry": ry,
                    "score": score,
                }

                tracks.setdefault(frame, []).append(trk)

        return tracks

    def bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def filter_tracks_for_demo(self, tracks):
        # Clean demo filter:
        # For 0018, prioritize big objects/cars first, then pedestrians/cyclists.
        if not self.demo_filter:
            return tracks

        candidates = []

        for trk in tracks:
            if trk["score"] < self.min_score:
                continue

            if trk["z"] < 0.0 or trk["z"] > self.max_forward_m:
                continue

            if abs(trk["x"]) > self.max_side_m:
                continue

            x1, y1, x2, y2 = trk["bbox"]
            box_w = x2 - x1
            box_h = y2 - y1
            area = box_w * box_h

            if box_w < 15 or box_h < 15:
                continue

            # Ranking:
            # Car gets priority because professor expects large visible objects.
            class_bonus = 3000.0 if trk["cls"] == "Car" else 0.0
            rank = class_bonus + (trk["score"] * 1000.0) + (area * 0.03) - (trk["z"] * 1.5)

            candidates.append((rank, trk))

        candidates = sorted(candidates, key=lambda v: v[0], reverse=True)

        selected = []

        # First pass: choose cars
        for _, trk in candidates:
            if trk["cls"] != "Car":
                continue

            if any(self.bbox_iou(trk["bbox"], old["bbox"]) > 0.10 for old in selected):
                continue

            selected.append(trk)

            if len(selected) >= min(4, self.max_demo_tracks):
                break

        # Second pass: fill remaining with other useful tracks
        for _, trk in candidates:
            if len(selected) >= self.max_demo_tracks:
                break

            if trk in selected:
                continue

            if any(self.bbox_iou(trk["bbox"], old["bbox"]) > 0.10 for old in selected):
                continue

            selected.append(trk)

        return selected

    def make_cloud_msg(self, points, header):
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = np.asarray(points, dtype=np.float32).tobytes()
        return msg

    def cv2_to_image_msg(self, img, header):
        img = np.ascontiguousarray(img)
        msg = Image()
        msg.header = header
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        return msg

    def camera_box_corners(self, trk):
        h, w, l = trk["h"], trk["w"], trk["l"]
        x, y, z = trk["x"], trk["y"], trk["z"]
        ry = trk["ry"]

        x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
        y_c = np.array([ 0.0,  0.0,  0.0,  0.0, -h,   -h,   -h,   -h])
        z_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])

        R = np.array([
            [ math.cos(ry), 0.0, math.sin(ry)],
            [ 0.0,          1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)]
        ])

        corners = R @ np.vstack([x_c, y_c, z_c])
        corners += np.array([[x], [y], [z]])
        return corners.T

    def cam_points_to_velo(self, pts_cam):
        pts_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
        pts_velo = (self.cam_to_velo @ pts_h.T).T[:, :3]
        return pts_velo

    def publish_markers(self, header, frame_number):
        tracks = self.tracks_by_frame.get(frame_number, [])
        tracks = self.filter_tracks_for_demo(tracks)

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for i, trk in enumerate(tracks):
            r, g, b = self.class_color(trk["cls"])

            corners_cam = self.camera_box_corners(trk)
            corners_velo = self.cam_points_to_velo(corners_cam)

            marker = Marker()
            marker.header = header
            marker.ns = "tracking_boxes"
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.08
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0

            for a, c in edges:
                pa = corners_velo[a]
                pc = corners_velo[c]
                marker.points.append(Point(x=float(pa[0]), y=float(pa[1]), z=float(pa[2])))
                marker.points.append(Point(x=float(pc[0]), y=float(pc[1]), z=float(pc[2])))

            marker_array.markers.append(marker)

            top_center = corners_velo[4:8].mean(axis=0)
            text = Marker()
            text.header = header
            text.ns = "tracking_ids"
            text.id = 10000 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(top_center[0])
            text.pose.position.y = float(top_center[1])
            text.pose.position.z = float(top_center[2] + 0.8)
            text.scale.z = 0.40
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 1.0
            text.text = f"ID:{trk['track_id']}"

            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    def publish_images(self, header, frame_number):
        image_path = os.path.join(self.image_dir, f"{frame_number:06d}.png")
        img = cv2.imread(image_path)

        if img is None:
            return

        overlay = img.copy()

        tracks = self.tracks_by_frame.get(frame_number, [])
        tracks = self.filter_tracks_for_demo(tracks)

        for trk in tracks:
            x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
            r, g, b = self.class_color(trk["cls"])
            color = (int(b * 255), int(g * 255), int(r * 255))

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"ID:{trk['track_id']}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        self.image_pub.publish(self.cv2_to_image_msg(img, header))
        self.overlay_pub.publish(self.cv2_to_image_msg(overlay, header))

    def publish_frame(self):
        if not self.frames:
            return

        if self.frame_index >= len(self.frames):
            if self.loop:
                self.frame_index = 0
            else:
                return

        frame_number = self.frames[self.frame_index]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        lidar_path = os.path.join(self.velodyne_dir, f"{frame_number:06d}.bin")
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        if self.max_points > 0 and points.shape[0] > self.max_points:
            step = max(1, points.shape[0] // self.max_points)
            points = points[::step]

        self.points_pub.publish(self.make_cloud_msg(points, header))
        self.publish_markers(header, frame_number)
        self.publish_images(header, frame_number)

        self.frame_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = AB3DMOTTrackingRvizPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
