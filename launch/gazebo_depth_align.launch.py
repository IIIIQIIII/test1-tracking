import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory  # 添加这行
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

    # 返回一个包含两个 launch 的描述
    return LaunchDescription([
        gazebo_launch,
        depth_align_launch
    ])