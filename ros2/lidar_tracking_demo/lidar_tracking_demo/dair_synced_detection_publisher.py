#!/usr/bin/env python3

import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class DAIRSyncedDetectionPublisher(Node):
    def __init__(self):
        super().__init__("dair_synced_detection_publisher")

        self.declare_parameter(
            "dair_training_root",
            ""
        )
        self.declare_parameter("fps", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("max_points", 30000)
        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("max_boxes", 10)
        self.declare_parameter("min_lidar_points_in_box", 8)

        self.root = self.get_parameter("dair_training_root").value
        if not self.root:
            raise ValueError('Set dair_training_root to the prepared DAIR-V2X-I prediction/training folder.')
        self.fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.min_score = float(self.get_parameter("min_score").value)
        self.max_boxes = int(self.get_parameter("max_boxes").value)
        self.min_lidar_points_in_box = int(self.get_parameter("min_lidar_points_in_box").value)

        self.velo_dir = os.path.join(self.root, "velodyne")
        self.image_dir = os.path.join(self.root, "image_2")
        self.label_dir = os.path.join(self.root, "label_2")
        self.calib_dir = os.path.join(self.root, "calib")

        self.frames = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.velo_dir)
            if f.endswith(".bin")
        ])

        self.points_pub = self.create_publisher(PointCloud2, "/points_raw", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/dair_3d_boxes", 10)
        self.raw_image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.overlay_pub = self.create_publisher(Image, "/camera/dair_overlay", 10)

        self.index = 0

        self.get_logger().info(f"DAIR synced aligned root: {self.root}")
        self.get_logger().info(f"Frames: {len(self.frames)}")
        self.get_logger().info("Topics: /points_raw, /dair_3d_boxes, /camera/image_raw, /camera/dair_overlay")

        period = 1.0 / self.fps if self.fps > 0 else 1.0
        self.timer = self.create_timer(period, self.publish_frame)

    def read_points(self, frame_id):
        path = os.path.join(self.velo_dir, f"{frame_id}.bin")
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

        if self.max_points > 0 and len(pts) > self.max_points:
            idx = np.linspace(0, len(pts) - 1, self.max_points).astype(np.int64)
            pts = pts[idx]

        return pts

    def points_to_msg(self, pts, header):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = pts.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * pts.shape[0]
        msg.is_dense = True
        msg.data = np.ascontiguousarray(pts.astype(np.float32)).tobytes()
        return msg

    def find_image(self, frame_id):
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]:
            p = os.path.join(self.image_dir, frame_id + ext)
            if os.path.exists(p):
                return p
        return None

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

    def read_labels(self, frame_id):
        path = os.path.join(self.label_dir, f"{frame_id}.txt")
        labels = []

        if not os.path.exists(path):
            return labels

        for line in open(path, "r"):
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls = parts[0]
            if cls == "DontCare":
                continue

            try:
                score = float(parts[15]) if len(parts) > 15 else 1.0
                if score < self.min_score:
                    continue

                labels.append({
                    "cls": cls,
                    "bbox": [
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7])
                    ],
                    "score": score,
                })
            except Exception:
                continue

        labels.sort(key=lambda x: x["score"], reverse=True)

        if self.max_boxes > 0:
            labels = labels[:self.max_boxes]

        return labels

    def read_calib_projection(self, frame_id):
        path = os.path.join(self.calib_dir, f"{frame_id}.txt")
        if not os.path.exists(path):
            return None

        data = {}
        for line in open(path, "r"):
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in val.strip().split()], dtype=np.float32)

        if "P2" not in data:
            return None

        P2 = data["P2"].reshape(3, 4)

        R0 = data.get("R0_rect", data.get("R_rect", np.eye(3, dtype=np.float32).reshape(-1))).reshape(3, 3)
        Tr = data.get("Tr_velo_to_cam", data.get("Tr_velo_cam", data.get("Tr_velo2cam", None)))

        if Tr is None:
            return None

        Tr = Tr.reshape(3, 4)

        R0_4 = np.eye(4, dtype=np.float32)
        R0_4[:3, :3] = R0

        Tr_4 = np.eye(4, dtype=np.float32)
        Tr_4[:3, :4] = Tr

        velo_to_cam = R0_4 @ Tr_4
        return P2, velo_to_cam

    def project_lidar_to_image(self, pts, P2, velo_to_cam, img_shape):
        xyz = pts[:, :3]
        ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
        xyz_h = np.hstack([xyz, ones])

        cam = (velo_to_cam @ xyz_h.T).T
        z = cam[:, 2]

        front = z > 0.2

        cam_h = np.hstack([cam[:, :3], np.ones((cam.shape[0], 1), dtype=np.float32)])
        img_proj = (P2 @ cam_h.T).T

        denom = img_proj[:, 2]
        valid_denom = np.abs(denom) > 1e-6

        u = np.zeros_like(denom)
        v = np.zeros_like(denom)

        u[valid_denom] = img_proj[valid_denom, 0] / denom[valid_denom]
        v[valid_denom] = img_proj[valid_denom, 1] / denom[valid_denom]

        h, w = img_shape[:2]

        valid = (
            front
            & valid_denom
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
        )

        return u, v, valid

    def class_color_bgr(self, cls):
        if cls == "Car":
            return (0, 255, 0)
        if cls == "Pedestrian":
            return (0, 140, 255)
        if cls == "Cyclist":
            return (255, 80, 0)
        return (255, 255, 255)

    def class_color_rgb(self, cls):
        if cls == "Car":
            return (0.0, 1.0, 0.0)
        if cls == "Pedestrian":
            return (1.0, 0.45, 0.0)
        if cls == "Cyclist":
            return (0.0, 0.45, 1.0)
        return (1.0, 1.0, 1.0)

    def draw_overlay(self, img, labels):
        overlay = img.copy()

        for obj in labels:
            x1, y1, x2, y2 = obj["bbox"]
            color = self.class_color_bgr(obj["cls"])

            x1 = int(max(0, min(img.shape[1] - 1, x1)))
            x2 = int(max(0, min(img.shape[1] - 1, x2)))
            y1 = int(max(0, min(img.shape[0] - 1, y1)))
            y2 = int(max(0, min(img.shape[0] - 1, y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            txt = f"{obj['cls']} {obj['score']:.2f}"
            cv2.putText(
                overlay,
                txt,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

        return overlay

    def robust_lidar_box_from_points(self, obj, box_points):
        if len(box_points) < self.min_lidar_points_in_box:
            return None

        xyz = box_points[:, :3]

        center = np.median(xyz, axis=0)

        if obj["cls"] == "Car":
            max_dim = np.array([8.0, 4.0, 3.0])
            min_dim = np.array([1.2, 1.0, 1.0])
        elif obj["cls"] == "Pedestrian":
            max_dim = np.array([2.5, 2.5, 2.5])
            min_dim = np.array([0.4, 0.4, 1.2])
        else:
            max_dim = np.array([3.5, 2.5, 2.5])
            min_dim = np.array([0.8, 0.6, 1.0])

        # Keep only nearby cluster around median to avoid background points.
        keep = (
            (np.abs(xyz[:, 0] - center[0]) < max_dim[0] / 2.0)
            & (np.abs(xyz[:, 1] - center[1]) < max_dim[1] / 2.0)
            & (np.abs(xyz[:, 2] - center[2]) < max_dim[2] / 2.0)
        )

        xyz = xyz[keep]

        if len(xyz) < self.min_lidar_points_in_box:
            return None

        lo = np.quantile(xyz, 0.05, axis=0)
        hi = np.quantile(xyz, 0.95, axis=0)

        center = (lo + hi) / 2.0
        dim = hi - lo

        dim = np.maximum(dim, min_dim)
        dim = np.minimum(dim, max_dim)

        lo = center - dim / 2.0
        hi = center + dim / 2.0

        corners = np.array([
            [hi[0], hi[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [lo[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]],
        ], dtype=np.float32)

        return corners

    def make_box_marker(self, obj, corners, marker_id, header):
        marker = Marker()
        marker.header = header
        marker.ns = "dair_lidar_aligned_boxes"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.18

        r, g, b = self.class_color_rgb(obj["cls"])
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for a, b in edges:
            p1 = Point()
            p1.x = float(corners[a, 0])
            p1.y = float(corners[a, 1])
            p1.z = float(corners[a, 2])

            p2 = Point()
            p2.x = float(corners[b, 0])
            p2.y = float(corners[b, 1])
            p2.z = float(corners[b, 2])

            marker.points.append(p1)
            marker.points.append(p2)

        return marker

    def make_text_marker(self, obj, corners, marker_id, header):
        center = corners.mean(axis=0)

        marker = Marker()
        marker.header = header
        marker.ns = "dair_lidar_aligned_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2] + 1.0)
        marker.scale.z = 0.7

        r, g, b = self.class_color_rgb(obj["cls"])
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        marker.text = f"{obj['cls']} {obj['score']:.2f}"
        return marker

    def publish_delete_all_markers(self, header):
        arr = MarkerArray()
        m = Marker()
        m.header = header
        m.action = Marker.DELETEALL
        arr.markers.append(m)
        self.marker_pub.publish(arr)

    def publish_frame(self):
        if not self.frames:
            return

        if self.index >= len(self.frames):
            if self.loop:
                self.index = 0
            else:
                return

        frame_id = self.frames[self.index]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "velodyne"

        pts = self.read_points(frame_id)
        labels = self.read_labels(frame_id)

        self.points_pub.publish(self.points_to_msg(pts, header))

        image_path = self.find_image(frame_id)
        img = None

        if image_path is not None:
            img = cv2.imread(image_path)

        if img is not None:
            img_header = Header()
            img_header.stamp = header.stamp
            img_header.frame_id = "camera"

            overlay = self.draw_overlay(img, labels)
            self.raw_image_pub.publish(self.cv2_to_image_msg(img, img_header))
            self.overlay_pub.publish(self.cv2_to_image_msg(overlay, img_header))

        self.publish_delete_all_markers(header)

        marker_array = MarkerArray()
        lidar_boxes = 0

        calib = None
        if img is not None:
            calib = self.read_calib_projection(frame_id)

        if calib is not None:
            P2, velo_to_cam = calib
            u, v, valid = self.project_lidar_to_image(pts, P2, velo_to_cam, img.shape)

            for i, obj in enumerate(labels):
                x1, y1, x2, y2 = obj["bbox"]

                mask = (
                    valid
                    & (u >= x1)
                    & (u <= x2)
                    & (v >= y1)
                    & (v <= y2)
                )

                selected_pts = pts[mask]

                corners = self.robust_lidar_box_from_points(obj, selected_pts)

                if corners is None:
                    continue

                marker_array.markers.append(self.make_box_marker(obj, corners, i, header))
                marker_array.markers.append(self.make_text_marker(obj, corners, 10000 + i, header))
                lidar_boxes += 1

        self.marker_pub.publish(marker_array)

        self.get_logger().info(
            f"Frame {frame_id}: points={len(pts)}, image_boxes={len(labels)}, lidar_boxes={lidar_boxes}",
            throttle_duration_sec=2.0
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DAIRSyncedDetectionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
