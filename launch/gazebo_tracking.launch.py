import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node  # 添加这行
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取当前工作空间中两个 launch 文件的路径
    go2_config_path = os.path.join(
        get_package_share_directory('go2_config'),
        'launch',
        'gazebo_mid360.launch.py'
    )

    depth_align_path = os.path.join(
        get_package_share_directory('depth_align_pkg'),
        'launch',
        'depth_align.launch.py'
    )

    # 包含第一个 launch 文件
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(go2_config_path),
        launch_arguments={'rviz': 'false'}.items()  # 设置 rviz 参数
    )

    # 包含第二个 launch 文件
    depth_align_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(depth_align_path)
    )

    # 添加 interactive_segmentation_node
    interactive_segmentation_node = Node(
        package='tracking',
        executable='interactive_segmentation',
        name='interactive_segmentation_node',
        output='screen'
    )

    # 添加 tracking_difference_node
    tracking_difference_node = Node(
        package='tracking',
        executable='tracking_difference',
        name='tracking_difference_node',
        output='screen'
    )

    # 返回一个包含所有节点和launch的描述
    return LaunchDescription([
        gazebo_launch,
        depth_align_launch,
        interactive_segmentation_node,
        tracking_difference_node
    ])