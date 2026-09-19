from glob import glob

from setuptools import find_packages, setup

package_name = 'lidar_tracking_demo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bhautik Zanzmera',
    maintainer_email='Bhautikzanzmera@users.noreply.github.com',
    description='ROS 2 visualization and integration for LiDAR 3D detection and tracking',
    license='MIT',
    entry_points={
        'console_scripts': [
            'kitti_lidar_publisher = lidar_tracking_demo.kitti_lidar_publisher:main',
            'kitti_lidar_box_publisher = lidar_tracking_demo.kitti_lidar_box_publisher:main',
            'ab3dmot_tracking_rviz_publisher = lidar_tracking_demo.ab3dmot_tracking_rviz_publisher:main',
            'dair_image_overlay_publisher = lidar_tracking_demo.dair_image_overlay_publisher:main',
            'dair_synced_detection_publisher = lidar_tracking_demo.dair_synced_detection_publisher:main',
            'dair_fast_tracking_publisher = lidar_tracking_demo.dair_fast_tracking_publisher:main',
            'dair_synced_tracking_publisher = lidar_tracking_demo.dair_synced_tracking_publisher:main',
            'kitti_tracking_rviz_publisher = lidar_tracking_demo.kitti_tracking_rviz_publisher:main',
        ],
    },
)
