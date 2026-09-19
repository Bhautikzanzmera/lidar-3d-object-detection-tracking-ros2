#!/usr/bin/env python3

import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image


class DAIRImageOverlayPublisher(Node):
    def __init__(self):
        super().__init__("dair_image_overlay_publisher")

        self.declare_parameter(
            "dair_training_root",
            ""
        )
        self.declare_parameter("fps", 1.0)
        self.declare_parameter("loop", True)

        self.root = self.get_parameter("dair_training_root").value
        if not self.root:
            raise ValueError('Set dair_training_root to the prepared DAIR-V2X-I KITTI-like training folder.')
        self.fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)

        self.image_dir = os.path.join(self.root, "image_2")
        self.label_dir = os.path.join(self.root, "label_2")

        self.frames = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.image_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ])

        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.overlay_pub = self.create_publisher(Image, "/camera/dair_overlay", 10)

        self.index = 0

        self.get_logger().info(f"DAIR image root: {self.root}")
        self.get_logger().info(f"Frames: {len(self.frames)}")
        self.get_logger().info("Publishing: /camera/image_raw and /camera/dair_overlay")

        period = 1.0 / self.fps if self.fps > 0 else 1.0
        self.timer = self.create_timer(period, self.publish_frame)

    def class_color(self, cls):
        # OpenCV uses BGR
        if cls == "Car":
            return (0, 255, 0)        # green
        if cls == "Pedestrian":
            return (0, 120, 255)      # orange
        if cls == "Cyclist":
            return (255, 80, 0)       # blue-ish
        return (255, 255, 255)

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

    def draw_labels(self, img, frame_id):
        label_path = os.path.join(self.label_dir, f"{frame_id}.txt")

        if not os.path.exists(label_path):
            return img

        overlay = img.copy()

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            # KITTI label format:
            # type trunc occl alpha x1 y1 x2 y2 h w l x y z ry
            if len(parts) < 15:
                continue

            cls = parts[0]
            if cls == "DontCare":
                continue

            try:
                x1 = int(float(parts[4]))
                y1 = int(float(parts[5]))
                x2 = int(float(parts[6]))
                y2 = int(float(parts[7]))
            except Exception:
                continue

            color = self.class_color(cls)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                overlay,
                cls,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
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
            else:
                return

        frame_id = self.frames[self.index]

        image_path_png = os.path.join(self.image_dir, f"{frame_id}.png")
        image_path_jpg = os.path.join(self.image_dir, f"{frame_id}.jpg")
        image_path_jpeg = os.path.join(self.image_dir, f"{frame_id}.jpeg")

        if os.path.exists(image_path_png):
            image_path = image_path_png
        elif os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        elif os.path.exists(image_path_jpeg):
            image_path = image_path_jpeg
        else:
            self.get_logger().warn(f"Image not found for frame {frame_id}")
            self.index += 1
            return

        img = cv2.imread(image_path)
        if img is None:
            self.index += 1
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera"

        overlay = self.draw_labels(img, frame_id)

        self.image_pub.publish(self.cv2_to_image_msg(img, header))
        self.overlay_pub.publish(self.cv2_to_image_msg(overlay, header))

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DAIRImageOverlayPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
