import os
import struct

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class KittiLidarPublisher(Node):
    def __init__(self):
        super().__init__('kitti_lidar_publisher')

        self.declare_parameter(
            'velodyne_dir',
            ''
        )
        self.declare_parameter('frame_id', 'velodyne')
        self.declare_parameter('fps', 2.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('max_points', 0)

        self.velodyne_dir = self.get_parameter('velodyne_dir').value
        self.frame_id = self.get_parameter('frame_id').value
        self.fps = float(self.get_parameter('fps').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.max_points = int(self.get_parameter('max_points').value)

        if not self.velodyne_dir:
            raise ValueError('Set velodyne_dir to a KITTI velodyne folder.')

        if self.fps <= 0:
            self.get_logger().warning('fps must be positive. Using fps=2.0')
            self.fps = 2.0

        if not os.path.isdir(self.velodyne_dir):
            raise FileNotFoundError(f'Velodyne directory not found: {self.velodyne_dir}')

        self.bin_files = sorted([
            f for f in os.listdir(self.velodyne_dir)
            if f.endswith('.bin')
        ])

        if not self.bin_files:
            raise RuntimeError(f'No .bin files found in: {self.velodyne_dir}')

        self.publisher_ = self.create_publisher(
            PointCloud2,
            '/points_raw',
            10
        )

        self.index = 0

        self.get_logger().info(f'Velodyne directory: {self.velodyne_dir}')
        self.get_logger().info(f'Number of LiDAR frames: {len(self.bin_files)}')
        self.get_logger().info(f'Frame ID: {self.frame_id}')
        self.get_logger().info(f'FPS: {self.fps}')
        self.get_logger().info(f'Loop: {self.loop}')
        self.get_logger().info(f'Max points: {self.max_points}')

        self.timer = self.create_timer(1.0 / self.fps, self.publish_next_cloud)

    def read_kitti_bin(self, file_path):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        if self.max_points > 0 and points.shape[0] > self.max_points:
            points = points[:self.max_points]

        return points

    def points_to_pointcloud2(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
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

    def publish_next_cloud(self):
        if self.index >= len(self.bin_files):
            if self.loop:
                self.get_logger().info('Reached end of LiDAR frames. Restarting playback.')
                self.index = 0
            else:
                self.get_logger().info('Finished LiDAR playback.')
                rclpy.shutdown()
                return

        bin_name = self.bin_files[self.index]
        bin_path = os.path.join(self.velodyne_dir, bin_name)

        points = self.read_kitti_bin(bin_path)
        msg = self.points_to_pointcloud2(points)

        self.publisher_.publish(msg)

        frame_num = os.path.splitext(bin_name)[0]
        self.get_logger().info(
            f'Published LiDAR frame {frame_num}: {points.shape[0]} points'
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)

    node = KittiLidarPublisher()
    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()
