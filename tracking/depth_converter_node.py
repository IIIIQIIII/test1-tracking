#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
from tf_transformations import quaternion_matrix

class DepthConverter(Node):
    def __init__(self):
        super().__init__('depth_converter')
        
        # 订阅深度图像
        self.depth_sub = self.create_subscription(
            Image,
            '/aligned_depth/image_raw',
            self.depth_callback,
            10)
            
        # 订阅odom信息
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom/ground_truth',
            self.odom_callback,
            10)
            
        self.bridge = CvBridge()
        
        # 相机内参
        self.K = np.array([
            [462.1379699707031, 0, 320.0],
            [0, 462.1379699707031, 240.0],
            [0, 0, 1]
        ])
        
        # 相机到base_link的固定变换
        self.camera_to_base = np.eye(4)
        # 根据tf2_echo的结果设置旋转矩阵
        self.camera_to_base[:3, :3] = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ])
        # 设置平移向量
        self.camera_to_base[:3, 3] = np.array([0.310, 0.033, 0.083])
        
        self.latest_odom = None

    def odom_callback(self, msg):
        self.latest_odom = msg

    def depth_callback(self, msg):
        if self.latest_odom is None:
            return
            
        # 转换深度图像
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        
        # 获取图像中心点的深度值(单位:米)
        center_depth = depth_image[240, 320] / 1000.0  # 转换为米
        
        # 计算相机坐标系下的3D点
        center_point_camera = np.array([
            (320 - self.K[0,2]) * center_depth / self.K[0,0],
            (240 - self.K[1,2]) * center_depth / self.K[1,1],
            center_depth,
            1.0  # 齐次坐标
        ])
        
        # 转换到base_link坐标系
        point_base = self.camera_to_base @ center_point_camera
        
        # 从odom中获取base到world的变换
        base_to_world = self.get_transform_matrix(self.latest_odom.pose.pose)
        
        # 转换到world坐标系
        point_world = base_to_world @ point_base
        
        self.get_logger().info(f'World coordinates: x={point_world[0]:.3f}, y={point_world[1]:.3f}, z={point_world[2]:.3f}')

    def get_transform_matrix(self, pose):
        # 从pose消息创建变换矩阵
        transform = np.eye(4)
        
        # 提取四元数
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        
        # 转换四元数到旋转矩阵
        transform[:3, :3] = quaternion_matrix(q)[:3, :3]
        
        # 设置平移向量
        transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        
        return transform

def main(args=None):
    rclpy.init(args=args)
    node = DepthConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()