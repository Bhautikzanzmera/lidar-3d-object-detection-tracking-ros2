import math
import os

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class KittiTrackingRvizPublisher(Node):
    def __init__(self):
        super().__init__('kitti_tracking_rviz_publisher')

        default_root = ''

        self.declare_parameter('kitti_tracking_root', default_root)
        self.declare_parameter('sequence', '0000')
        self.declare_parameter('frame_id', 'velodyne')
        self.declare_parameter('fps', 5.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('max_points', 30000)

        self.kitti_root = self.get_parameter('kitti_tracking_root').value
        self.sequence = str(self.get_parameter('sequence').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.fps = float(self.get_parameter('fps').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.max_points = int(self.get_parameter('max_points').value)

        if not self.kitti_root:
            raise ValueError('Set kitti_tracking_root to your KITTI tracking training folder.')

        if self.fps <= 0:
            self.get_logger().warning('fps must be positive. Using fps=5.0')
            self.fps = 5.0

        self.velodyne_dir = os.path.join(self.kitti_root, 'velodyne', self.sequence)
        self.label_path = os.path.join(self.kitti_root, 'label_02', f'{self.sequence}.txt')
        self.calib_path = os.path.join(self.kitti_root, 'calib', f'{self.sequence}.txt')

        for path in [self.velodyne_dir]:
            if not os.path.isdir(path):
                raise FileNotFoundError(f'Required directory not found: {path}')

        for path in [self.label_path, self.calib_path]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Required file not found: {path}')

        self.bin_files = sorted([
            f for f in os.listdir(self.velodyne_dir)
            if f.endswith('.bin')
        ])

        if not self.bin_files:
            raise RuntimeError(f'No .bin files found in {self.velodyne_dir}')

        self.cam_rect_to_velo = self.read_calib(self.calib_path)
        self.tracks_by_frame = self.read_tracking_labels(self.label_path)

        self.cloud_pub = self.create_publisher(PointCloud2, '/points_raw', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/kitti_tracking_3d_boxes', 10)

        self.index = 0

        self.get_logger().info(f'KITTI tracking root: {self.kitti_root}')
        self.get_logger().info(f'Sequence: {self.sequence}')
        self.get_logger().info(f'Number of LiDAR frames: {len(self.bin_files)}')
        self.get_logger().info(f'Label file: {self.label_path}')
        self.get_logger().info(f'Calibration file: {self.calib_path}')
        self.get_logger().info(f'Frame ID: {self.frame_id}')
        self.get_logger().info(f'FPS: {self.fps}')
        self.get_logger().info(f'Loop: {self.loop}')
        self.get_logger().info(f'Max points: {self.max_points}')

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
                line = line.strip()

                if not line:
                    continue

                # KITTI files may use either:
                # R_rect: values
                # or
                # R_rect values
                if ':' in line:
                    key, value = line.split(':', 1)
                    values = value.strip().split()
                else:
                    parts = line.split()
                    key = parts[0]
                    values = parts[1:]

                if not values:
                    continue

                data[key] = np.array([float(x) for x in values])

        # KITTI tracking commonly uses R_rect and Tr_velo_cam.
        # KITTI object detection may use R0_rect and Tr_velo_to_cam.
        r_key_candidates = ['R_rect', 'R_rect_00', 'R0_rect']
        tr_key_candidates = ['Tr_velo_cam', 'Tr_velo_to_cam', 'Tr_velo_to_cam_00']

        r_key = next((k for k in r_key_candidates if k in data), None)
        tr_key = next((k for k in tr_key_candidates if k in data), None)

        if r_key is None:
            raise KeyError(f'Could not find rectification matrix in {calib_path}. Available keys: {list(data.keys())}')

        if tr_key is None:
            raise KeyError(f'Could not find Velodyne-to-camera transform in {calib_path}. Available keys: {list(data.keys())}')

        r_rect = np.eye(4)
        r_rect[:3, :3] = data[r_key].reshape(3, 3)

        tr_velo_cam = np.eye(4)
        tr_velo_cam[:3, :4] = data[tr_key].reshape(3, 4)

        cam_rect_to_velo = np.linalg.inv(r_rect @ tr_velo_cam)

        return cam_rect_to_velo

    def read_tracking_labels(self, label_path):
        tracks_by_frame = {}

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 17:
                    continue

                frame = int(parts[0])
                track_id = int(parts[1])
                obj_class = parts[2]

                if obj_class == 'DontCare' or track_id < 0:
                    continue

                h = float(parts[10])
                w = float(parts[11])
                l = float(parts[12])

                x = float(parts[13])
                y = float(parts[14])
                z = float(parts[15])

                ry = float(parts[16])

                item = {
                    'frame': frame,
                    'track_id': track_id,
                    'class': obj_class,
                    'h': h,
                    'w': w,
                    'l': l,
                    'location': np.array([x, y, z], dtype=np.float32),
                    'rotation_y': ry,
                }

                tracks_by_frame.setdefault(frame, []).append(item)

        return tracks_by_frame

    def compute_box_corners_camera(self, obj):
        h = obj['h']
        w = obj['w']
        l = obj['l']
        x, y, z = obj['location']
        ry = obj['rotation_y']

        # KITTI tracking labels store 3D boxes in camera coordinates.
        # Location is bottom-center of the object box.
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

    def camera_corners_to_velodyne(self, corners_camera):
        ones = np.ones((corners_camera.shape[0], 1))
        corners_hom = np.hstack([corners_camera, ones])
        corners_velo = (self.cam_rect_to_velo @ corners_hom.T).T[:, :3]
        return corners_velo

    def make_box_marker(self, corners, marker_id, obj, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = 'kitti_tracking_3d_boxes'
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.08

        # Green boxes.
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

    def make_text_marker(self, corners, marker_id, obj, stamp):
        center = np.mean(corners, axis=0)

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = 'kitti_tracking_labels'
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

        marker.text = f"{obj['class']} ID:{obj['track_id']}"
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
                self.get_logger().info('Reached end of tracking sequence. Restarting playback.')
                self.index = 0
            else:
                self.get_logger().info('Finished tracking sequence.')
                rclpy.shutdown()
                return

        bin_name = self.bin_files[self.index]
        frame_name = os.path.splitext(bin_name)[0]
        frame_number = int(frame_name)

        bin_path = os.path.join(self.velodyne_dir, bin_name)

        stamp = self.get_clock().now().to_msg()

        points = self.read_kitti_bin(bin_path)
        cloud_msg = self.points_to_pointcloud2(points, stamp)
        self.cloud_pub.publish(cloud_msg)

        marker_array = MarkerArray()
        marker_array.markers.append(self.make_delete_all_marker(stamp))

        objects = self.tracks_by_frame.get(frame_number, [])

        marker_id = 0
        for obj in objects:
            corners_camera = self.compute_box_corners_camera(obj)
            corners_velo = self.camera_corners_to_velodyne(corners_camera)

            marker_array.markers.append(
                self.make_box_marker(corners_velo, marker_id, obj, stamp)
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(corners_velo, marker_id, obj, stamp)
            )
            marker_id += 1

        self.marker_pub.publish(marker_array)

        self.get_logger().info(
            f'Seq {self.sequence}, frame {frame_name}: '
            f'published {points.shape[0]} points and {len(objects)} tracked boxes'
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)

    node = KittiTrackingRvizPublisher()
    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()
