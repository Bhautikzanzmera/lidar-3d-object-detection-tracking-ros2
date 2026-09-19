from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    sequence = LaunchConfiguration('sequence')
    fps = LaunchConfiguration('fps')
    max_points = LaunchConfiguration('max_points')
    kitti_tracking_root = LaunchConfiguration('kitti_tracking_root')

    return LaunchDescription([
        DeclareLaunchArgument(
            'sequence',
            default_value='0000',
            description='KITTI tracking sequence, for example 0000 or 0001'
        ),

        DeclareLaunchArgument(
            'fps',
            default_value='5.0',
            description='Playback speed in frames per second'
        ),

        DeclareLaunchArgument(
            'max_points',
            default_value='30000',
            description='Maximum LiDAR points published per frame'
        ),

        DeclareLaunchArgument(
            'kitti_tracking_root',
            default_value='',
            description='Path to the KITTI tracking training folder (required)'
        ),

        Node(
            package='lidar_tracking_demo',
            executable='kitti_tracking_rviz_publisher',
            name='kitti_tracking_rviz_publisher',
            output='screen',
            parameters=[{
                'sequence': ParameterValue(sequence, value_type=str),
                'fps': ParameterValue(fps, value_type=float),
                'max_points': ParameterValue(max_points, value_type=int),
                'kitti_tracking_root': ParameterValue(kitti_tracking_root, value_type=str),
            }]
        )
    ])
