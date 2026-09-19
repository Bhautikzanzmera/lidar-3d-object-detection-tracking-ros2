import math
import os

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class KittiLidarBoxPublisher(Node):
    def __init__(self):
        super().__init__('kitti_lidar_box_publisher')

        default_root = ''

        self.declare_parameter('kitti_training_root', default_root)
        self.declare_parameter('frame_id', 'velodyne')
        self.declare_parameter('fps', 1.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('max_points', 30000)
        self.declare_parameter('label_coord_frame', 'camera')

        self.kitti_root = self.get_parameter('kitti_training_root').value
        self.frame_id = self.get_parameter('frame_id').value
        self.fps = float(self.get_parameter('fps').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.label_coord_frame = str(self.get_parameter('label_coord_frame').value).lower()

        if not self.kitti_root:
            raise ValueError('Set kitti_training_root to your KITTI object-detection training folder.')

        if self.fps <= 0:
            self.get_logger().warning('fps must be positive. Using fps=1.0')
            self.fps = 1.0

        self.velodyne_dir = os.path.join(self.kitti_root, 'velodyne')
        self.label_dir = os.path.join(self.kitti_root, 'label_2')
        self.calib_dir = os.path.join(self.kitti_root, 'calib')

        for path in [self.velodyne_dir, self.label_dir, self.calib_dir]:
            if not os.path.isdir(path):
                raise FileNotFoundError(f'Required KITTI directory not found: {path}')

        self.bin_files = sorted([
            f for f in os.listdir(self.velodyne_dir)
            if f.endswith('.bin')
        ])

        if not self.bin_files:
            raise RuntimeError(f'No .bin files found in {self.velodyne_dir}')

        self.cloud_pub = self.create_publisher(PointCloud2, '/points_raw', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/kitti_3d_boxes', 10)

        self.index = 0

        self.get_logger().info(f'KITTI training root: {self.kitti_root}')
        self.get_logger().info(f'Number of LiDAR frames: {len(self.bin_files)}')
        self.get_logger().info(f'Frame ID: {self.frame_id}')
        self.get_logger().info(f'FPS: {self.fps}')
        self.get_logger().info(f'Loop: {self.loop}')
        self.get_logger().info(f'Max points: {self.max_points}')
        self.get_logger().info(f'Label coordinate frame: {self.label_coord_frame}')

        self.timer = self.create_timer(1.0 / self.fps, self.publish_next_frame)

    def read_kitti_bin(self, file_path):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        if self.max_points > 0 and points.shape[0] > self.max_points:
            points = points[:self.max_points]

        return points

    def points_to_pointcloud2(self, points, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()

        return msg

    def read_calib(self, calib_path):
        data = {}

        with open(calib_path, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue

                key, value = line.split(':', 1)
                values = np.array([float(x) for x in value.strip().split()])

                if key == 'R0_rect':
                    data[key] = values.reshape(3, 3)
                elif key == 'Tr_velo_to_cam':
                    data[key] = values.reshape(3, 4)

        r0 = np.eye(4)
        r0[:3, :3] = data['R0_rect']

        tr = np.eye(4)
        tr[:3, :4] = data['Tr_velo_to_cam']

        cam_rect_to_velo = np.linalg.inv(r0 @ tr)

        return cam_rect_to_velo

    def read_labels(self, label_path):
        objects = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 15:
                    continue

                obj_class = parts[0]

                if obj_class == 'DontCare':
                    continue

                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])

                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])

                ry = float(parts[14])

                objects.append({
                    'class': obj_class,
                    'h': h,
                    'w': w,
                    'l': l,
                    'location': np.array([x, y, z], dtype=np.float32),
                    'rotation_y': ry,
                })

        return objects

    def compute_box_corners_camera(self, obj):
        h = obj['h']
        w = obj['w']
        l = obj['l']
        x, y, z = obj['location']
        ry = obj['rotation_y']

        # KITTI object location is bottom-center in camera coordinates.
        x_corners = np.array([
            l / 2, l / 2, -l / 2, -l / 2,
            l / 2, l / 2, -l / 2, -l / 2
        ])

        y_corners = np.array([
            0, 0, 0, 0,
            -h, -h, -h, -h
        ])

        z_corners = np.array([
            w / 2, -w / 2, -w / 2, w / 2,
            w / 2, -w / 2, -w / 2, w / 2
        ])

        rotation = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])

        corners = rotation @ np.vstack([x_corners, y_corners, z_corners])
        corners = corners + np.array([[x], [y], [z]])

        return corners.T

    def camera_corners_to_velodyne(self, corners_camera, cam_rect_to_velo):
        ones = np.ones((corners_camera.shape[0], 1))
        corners_hom = np.hstack([corners_camera, ones])
        corners_velo = (cam_rect_to_velo @ corners_hom.T).T[:, :3]
        return corners_velo

    def compute_box_corners_velodyne(self, obj):
        h = obj['h']
        w = obj['w']
        l = obj['l']
        x, y, z = obj['location']
        yaw = obj['rotation_y']

        x_corners = np.array([
            l / 2, l / 2, -l / 2, -l / 2,
            l / 2, l / 2, -l / 2, -l / 2
        ])

        y_corners = np.array([
            w / 2, -w / 2, -w / 2, w / 2,
            w / 2, -w / 2, -w / 2, w / 2
        ])

        z_corners = np.array([
            -h / 2, -h / 2, -h / 2, -h / 2,
            h / 2, h / 2, h / 2, h / 2
        ])

        rotation = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        corners = rotation @ np.vstack([x_corners, y_corners, z_corners])
        corners = corners + np.array([[x], [y], [z]])

        return corners.T

    def make_box_marker(self, corners, marker_id, obj_class, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = 'kitti_3d_boxes'
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.08

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for start, end in edges:
            p1 = Point()
            p1.x, p1.y, p1.z = corners[start].tolist()

            p2 = Point()
            p2.x, p2.y, p2.z = corners[end].tolist()

            marker.points.append(p1)
            marker.points.append(p2)

        marker.lifetime.sec = 1

        return marker

    def make_text_marker(self, corners, marker_id, obj_class, stamp):
        center = np.mean(corners, axis=0)

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = 'kitti_3d_box_labels'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2] + 1.0)

        marker.scale.z = 0.8

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.text = obj_class
        marker.lifetime.sec = 1

        return marker

    def make_delete_all_marker(self, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        return marker

    def publish_next_frame(self):
        if self.index >= len(self.bin_files):
            if self.loop:
                self.get_logger().info('Reached end of KITTI frames. Restarting playback.')
                self.index = 0
            else:
                self.get_logger().info('Finished KITTI playback.')
                rclpy.shutdown()
                return

        bin_name = self.bin_files[self.index]
        frame_name = os.path.splitext(bin_name)[0]

        bin_path = os.path.join(self.velodyne_dir, f'{frame_name}.bin')
        label_path = os.path.join(self.label_dir, f'{frame_name}.txt')
        calib_path = os.path.join(self.calib_dir, f'{frame_name}.txt')

        stamp = self.get_clock().now().to_msg()

        points = self.read_kitti_bin(bin_path)
        cloud_msg = self.points_to_pointcloud2(points, stamp)
        self.cloud_pub.publish(cloud_msg)

        marker_array = MarkerArray()
        marker_array.markers.append(self.make_delete_all_marker(stamp))

        object_count = 0

        if os.path.isfile(label_path) and os.path.isfile(calib_path):
            cam_rect_to_velo = self.read_calib(calib_path)
            objects = self.read_labels(label_path)

            marker_id = 0
            for obj in objects:
                if self.label_coord_frame == 'camera':
                    corners_camera = self.compute_box_corners_camera(obj)
                    corners_velo = self.camera_corners_to_velodyne(
                        corners_camera,
                        cam_rect_to_velo
                    )
                elif self.label_coord_frame in ['velodyne', 'lidar']:
                    corners_velo = self.compute_box_corners_velodyne(obj)
                else:
                    self.get_logger().warning(
                        f"Unknown label_coord_frame={self.label_coord_frame}; skipping box"
                    )
                    continue

                marker_array.markers.append(
                    self.make_box_marker(corners_velo, marker_id, obj['class'], stamp)
                )
                marker_id += 1

                label_text = f"{obj['class']} #{object_count}"
                marker_array.markers.append(
                    self.make_text_marker(corners_velo, marker_id, label_text, stamp)
                )
                marker_id += 1

                object_count += 1

        self.marker_pub.publish(marker_array)

        self.get_logger().info(
            f'Frame {frame_name}: published {points.shape[0]} points and {object_count} boxes'
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)

    node = KittiLidarBoxPublisher()
    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()
