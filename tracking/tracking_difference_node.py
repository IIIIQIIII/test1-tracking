#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf_transformations import quaternion_matrix

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')
        self.bridge = CvBridge()

        # Subscribers
        self.mask_sub = self.create_subscription(
            Image,
            '/segmentation_mask',
            self.mask_callback,
            10)

        self.depth_sub = self.create_subscription(
            Image,
            '/aligned_depth/image_raw',
            self.depth_callback,
            10)
            
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom/ground_truth',
            self.odom_callback,
            10)
        
        # Publishers
        self.position_pub = self.create_publisher(PointStamped, '/tracking_object/position', 10)
        self.velocity_pub = self.create_publisher(TwistStamped, '/tracking_object/velocity', 10)

        # Camera intrinsics
        self.K = np.array([
            [462.1379699707031, 0, 320.0],
            [0, 462.1379699707031, 240.0],
            [0, 0, 1]
        ])
        
        # Camera to base_link transform
        self.camera_to_base = np.eye(4)
        self.camera_to_base[:3, :3] = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ])
        self.camera_to_base[:3, 3] = np.array([0.310, 0.033, 0.083])

        # Tracking variables
        self.track_id = 1
        self.track_start_time = None
        self.prev_centroid = None
        self.prev_time = None
        self.smoothed_velocity = np.zeros(3)
        self.velocity_alpha = 0.3  # Velocity smoothing factor

        self.depth_image = None
        self.mask_image = None
        self.latest_odom = None

        self.get_logger().info("Object Tracking Node Initialized")

    def odom_callback(self, msg):
        self.latest_odom = msg

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))

    def mask_callback(self, msg):
        try:
            self.mask_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error('Could not convert mask image: %s' % str(e))
            return

        if self.depth_image is not None and self.mask_image is not None and self.latest_odom is not None:
            self.process_mask()

    def process_mask(self):
        mask = self.mask_image
        depth_image = self.depth_image
        
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0:
            self.get_logger().warn("No object detected in the mask.")
            return

        # Initialize tracking if not started
        if self.track_start_time is None:
            self.track_start_time = self.get_clock().now()

        object_points_camera = []
        for x, y in zip(x_indices, y_indices):
            depth = depth_image[y, x] / 1000.0  # Convert to meters
            if np.isnan(depth) or np.isinf(depth) or depth <= 0:
                continue

            point_camera = self.pixel_to_camera(x, y, depth)
            object_points_camera.append(point_camera)

        if not object_points_camera:
            self.get_logger().warn("No valid depth data found within the mask.")
            return

        centroid_camera = np.mean(object_points_camera, axis=0)
        centroid_world = self.transform_to_world(centroid_camera)
        
        current_time = self.get_clock().now()
        
        # Calculate velocity
        velocity = np.zeros(3)
        if self.prev_centroid is not None and self.prev_time is not None:
            dt = (current_time.nanoseconds - self.prev_time.nanoseconds) / 1e9
            if dt > 0:
                velocity = (centroid_world - self.prev_centroid) / dt
                self.smoothed_velocity = self.velocity_alpha * velocity + \
                                      (1 - self.velocity_alpha) * self.smoothed_velocity

        # Calculate tracking duration
        tracking_duration = (current_time - self.track_start_time).nanoseconds / 1e9

        # Publish tracking message
        self.publish_tracking(centroid_world, self.smoothed_velocity)
        
        # Update previous values
        self.prev_centroid = centroid_world
        self.prev_time = current_time

    def publish_tracking(self, position, velocity):
        # Publish position
        pos_msg = PointStamped()
        pos_msg.header.stamp = self.get_clock().now().to_msg()
        pos_msg.header.frame_id = "world"
        pos_msg.point.x = float(position[0])
        pos_msg.point.y = float(position[1])
        pos_msg.point.z = float(position[2])
        
        # Publish velocity
        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = "world"
        vel_msg.twist.linear.x = float(velocity[0])
        vel_msg.twist.linear.y = float(velocity[1])
        vel_msg.twist.linear.z = float(velocity[2])
        
        self.position_pub.publish(pos_msg)
        self.velocity_pub.publish(vel_msg)
        
        self.get_logger().info(
            f"Published tracking: "
            f"pos=({pos_msg.point.x:.3f}, {pos_msg.point.y:.3f}, {pos_msg.point.z:.3f}), "
            f"vel=({vel_msg.twist.linear.x:.3f}, {vel_msg.twist.linear.y:.3f}, {vel_msg.twist.linear.z:.3f})"
        )

    def pixel_to_camera(self, u, v, depth):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.array([x, y, z])
        
    def transform_to_world(self, point_camera):
        point_camera_homogeneous = np.append(point_camera, 1.0)
        point_base = self.camera_to_base @ point_camera_homogeneous
        base_to_world = self.get_transform_matrix(self.latest_odom.pose.pose)
        point_world = base_to_world @ point_base
        return point_world[:3]

    def get_transform_matrix(self, pose):
        transform = np.eye(4)
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        transform[:3, :3] = quaternion_matrix(q)[:3, :3]
        transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return transform

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()