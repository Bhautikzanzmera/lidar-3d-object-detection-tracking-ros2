# ROS 2 LiDAR Tracking Demo

This ROS 2 package contains the visualization and integration nodes used in the LiDAR 3D detection and multi-object tracking portfolio project.

## Main capabilities

- Publish KITTI LiDAR point clouds as `sensor_msgs/PointCloud2`
- Visualize KITTI 3D boxes and tracking IDs in RViz2
- Visualize AB3DMOT tracking outputs with synchronized camera overlays
- Publish synchronized DAIR-V2X-I LiDAR/camera detection outputs

## Build

Place this package inside a ROS 2 workspace and build it with:

```bash
colcon build --packages-select lidar_tracking_demo
source install/setup.bash
```

## KITTI tracking visualization

Dataset paths are intentionally **not hard-coded**. Pass your local KITTI directory at runtime:

```bash
ros2 launch lidar_tracking_demo kitti_tracking_demo.launch.py \
  kitti_tracking_root:=/path/to/kitti_tracking/training \
  sequence:=0019 \
  fps:=5.0
```

For the AB3DMOT visualization node, also provide the directory containing tracking result files:

```bash
ros2 run lidar_tracking_demo ab3dmot_tracking_rviz_publisher --ros-args \
  -p kitti_tracking_root:=/path/to/kitti_tracking/training \
  -p ab3dmot_result_root:=/path/to/tracking/results \
  -p sequence:=0019
```

If a converted calibration file was used during detection/tracking conversion, pass it with:

```bash
-p calib_file:=/path/to/0019.txt
```

## Notes

Raw KITTI/DAIR-V2X data and AB3DMOT source code are not included in this repository. Obtain them from their official/upstream sources and follow their respective licenses and terms.
