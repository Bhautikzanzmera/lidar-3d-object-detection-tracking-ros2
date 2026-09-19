#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from lidar_tracking_demo.dair_synced_detection_publisher import DAIRSyncedDetectionPublisher


class DAIRFastTrackingPublisher(DAIRSyncedDetectionPublisher):
    def __init__(self):
        super().__init__()

        # Replace detection marker topic with tracking marker topic
        self.marker_pub = self.create_publisher(MarkerArray, "/dair_tracking_boxes", 10)

        self.declare_parameter("target_class", "Car")
        self.declare_parameter("max_track_distance", 6.0)

        self.target_class = str(self.get_parameter("target_class").value)
        self.max_track_distance = float(self.get_parameter("max_track_distance").value)

        self.prev_center = None
        self.track_id = 1

        self.get_logger().info("DAIR fast single-object tracking publisher started")
        self.get_logger().info("Tracking topic: /dair_tracking_boxes")
        self.get_logger().info(f"Target class: {self.target_class}")

    def choose_target(self, candidates):
        if len(candidates) == 0:
            return None

        # First frame: choose highest score object
        if self.prev_center is None:
            best = max(candidates, key=lambda c: c["score"])
            self.prev_center = best["center"]
            return best

        # Next frames: choose nearest object to previous tracked center
        best = None
        best_dist = 1e9

        for c in candidates:
            dist = float(np.linalg.norm(c["center"][:2] - self.prev_center[:2]))
            if dist < best_dist:
                best_dist = dist
                best = c

        # If distance is reasonable, track nearest object.
        # If object is temporarily unstable, still reacquire best-score car with same ID for demo.
        if best is not None and best_dist <= self.max_track_distance:
            self.prev_center = best["center"]
            return best

        best = max(candidates, key=lambda c: c["score"])
        self.prev_center = best["center"]
        return best

    def make_tracking_text_marker(self, obj, corners, marker_id, header):
        center = corners.mean(axis=0)

        marker = Marker()
        marker.header = header
        marker.ns = "dair_tracking_id_text"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2] + 1.2)

        marker.scale.z = 0.9

        r, g, b = self.class_color_rgb(obj["cls"])
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.text = f"ID:{self.track_id} {obj['cls']} {obj['score']:.2f}"
        return marker

    def draw_tracking_overlay(self, img, labels, target_obj):
        overlay = self.draw_overlay(img, labels)

        if target_obj is not None:
            x1, y1, x2, y2 = target_obj["bbox"]
            x1 = int(max(0, min(img.shape[1] - 1, x1)))
            x2 = int(max(0, min(img.shape[1] - 1, x2)))
            y1 = int(max(0, min(img.shape[0] - 1, y1)))
            y2 = int(max(0, min(img.shape[0] - 1, y2)))

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                overlay,
                f"TRACK ID:{self.track_id}",
                (x1, min(img.shape[0] - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        return overlay

    def publish_frame(self):
        if not self.frames:
            return

        if self.index >= len(self.frames):
            if self.loop:
                self.index = 0
                self.prev_center = None
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
        img = cv2.imread(image_path) if image_path is not None else None

        self.publish_delete_all_markers(header)

        marker_array = MarkerArray()
        selected_target = None
        candidates = []

        calib = None
        if img is not None:
            calib = self.read_calib_projection(frame_id)

        if calib is not None and img is not None:
            P2, velo_to_cam = calib
            u, v, valid = self.project_lidar_to_image(pts, P2, velo_to_cam, img.shape)

            # First try target class, normally Car
            search_labels = [obj for obj in labels if obj["cls"] == self.target_class]

            # If no target class exists in this frame, fallback to all objects
            if len(search_labels) == 0:
                search_labels = labels

            for obj in search_labels:
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

                candidates.append({
                    "obj": obj,
                    "corners": corners,
                    "center": corners.mean(axis=0),
                    "score": obj["score"],
                })

            target = self.choose_target(candidates)

            if target is not None:
                selected_target = target["obj"]
                selected_target["track_id"] = self.track_id

                marker_array.markers.append(
                    self.make_box_marker(selected_target, target["corners"], self.track_id, header)
                )
                marker_array.markers.append(
                    self.make_tracking_text_marker(selected_target, target["corners"], 10000 + self.track_id, header)
                )

        self.marker_pub.publish(marker_array)

        if img is not None:
            img_header = Header()
            img_header.stamp = header.stamp
            img_header.frame_id = "camera"

            overlay = self.draw_tracking_overlay(img, labels, selected_target)
            self.raw_image_pub.publish(self.cv2_to_image_msg(img, img_header))
            self.overlay_pub.publish(self.cv2_to_image_msg(overlay, img_header))

        tracked = 1 if selected_target is not None else 0
        self.get_logger().info(
            f"Frame {frame_id}: detections={len(labels)}, tracked_car={tracked}, ID={self.track_id}"
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DAIRFastTrackingPublisher()
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
