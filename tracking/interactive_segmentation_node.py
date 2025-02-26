import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import torch
import datetime
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
from sensor_msgs.msg import Image  # Import the Image message type
from std_msgs.msg import Header  # Import the Header message type

class InteractiveSegmentationNode(Node):

    def __init__(self):
        super().__init__('interactive_segmentation_node')
        self.subscription = self.create_subscription(
            Image,
            '/color/image_raw',  # Replace with your image topic
            self.image_callback,
            10)
        self.mask_publisher = self.create_publisher(
            Image,  # Publish masks as images
            'segmentation_mask',  # Topic name for masks
            10)
        self.bridge = CvBridge()
        self.frame = None
        self.segmentation_done = False
        self.clicked_points = []
        self.clicked_labels = []
        self.current_obj_id = 1
        self.predictor = None  # EfficientTAM predictor
        self.ann_frame_idx = 0 #annotation frame index

        # Load EfficientTAM (replace with your actual paths)
        checkpoint = "/home/newusername/real-time-eta/checkpoints/efficienttam_ti_512x512.pt"  # Replace with your checkpoint path
        model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"  # Replace with your config path
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint

        self.get_logger().info('Interactive Segmentation Node started.')

    def mouse_callback(self, event, x, y, flags, param):
        if not self.segmentation_done:  # Only allow clicks during annotation phase
            if event == cv2.EVENT_LBUTTONDOWN:  # Left click for positive
                self.clicked_points.append([x, y])
                self.clicked_labels.append(1)
            elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for negative
                self.clicked_points.append([x, y])
                self.clicked_labels.append(0)

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error('Error converting image: %s' % str(e))
            return

        if self.frame is not None:
            if not self.segmentation_done:
                self.annotation_phase(self.frame)
            else:
                self.track_objects(self.frame, msg.header)  # Pass the header

    def track_objects(self, frame, header): # Modified to accept the header
        # 跟踪阶段
        obj_ids, mask_logits = self.predictor.track(frame)

        # 创建叠加层
        overlay = frame.copy()
        alpha = 0.3  # 透明度

        # 预定义颜色表（BGR格式）
        colors = [
            (255, 0, 0),   # 红
            (0, 255, 0),   # 绿
            (0, 0, 255),   # 蓝
            (0, 255, 255), # 黄
            (255, 0, 255), # 品红
            (255, 255, 0), # 青
        ]

        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Initialize combined mask

        for idx, obj_id in enumerate(obj_ids):
            # 获取当前对象mask
            mask = (mask_logits[idx] > 0).squeeze().cpu().numpy().astype(np.uint8)
            color = colors[idx % len(colors)]

            # 绘制轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            # 半透明填充
            mask_color = np.zeros_like(overlay)
            mask_color[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, mask_color, 0.3, 0)

            # 计算中心点
            y, x = np.where(mask)
            if len(x) > 0:
                x_center, y_center = int(np.mean(x)), int(np.mean(y))
                cv2.circle(overlay, (x_center, y_center), 5, color, -1)
                cv2.putText(overlay, f'ID:{obj_id}', (x_center+5, y_center+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Combine masks (you might want to handle overlapping objects differently)
            combined_mask = cv2.bitwise_or(combined_mask, mask)  # Simple OR operation

        # 融合叠加层
        frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)

        cv2.imshow("Segmentation Result", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)  # Wait indefinitely until a key is pressed

        # Publish the combined mask
        self.publish_mask(combined_mask, header)

    def publish_mask(self, mask, header):
        """Publishes the segmentation mask as a ROS2 Image message."""
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")  # "mono8" for grayscale masks
            mask_msg.header = header  # Use the same header as the input image
            self.mask_publisher.publish(mask_msg)
            self.get_logger().info('Published segmentation mask')
        except Exception as e:
            self.get_logger().error('Error publishing mask: %s' % str(e))

    def annotation_phase(self, frame):
        if self.predictor is None:
            self.predictor = build_efficienttam_camera_predictor(self.model_cfg, self.checkpoint)
            self.predictor.load_first_frame(frame)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Image", self.mouse_callback)

        while not self.segmentation_done:
            display_frame = frame.copy()

            # Draw clicked points
            for p, l in zip(self.clicked_points, self.clicked_labels):
                color = (0, 255, 0) if l == 1 else (0, 0, 255)
                cv2.circle(display_frame, tuple(p), 5, color, -1)

            # Display object ID
            cv2.putText(display_frame, f"Object {self.current_obj_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Image", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):
                if self.clicked_points:
                    # Add object to tracker
                    points = np.array(self.clicked_points, dtype=np.float32)
                    labels = np.array(self.clicked_labels, dtype=np.int32)
                    self.predictor.add_new_prompt(
                        frame_idx=self.ann_frame_idx,
                        obj_id=self.current_obj_id,
                        points=points,
                        labels=labels
                    )
                    self.get_logger().info(f"Object {self.current_obj_id} added.")
                    self.current_obj_id += 1
                    self.clicked_points = []
                    self.clicked_labels = []
                else:
                    self.get_logger().warn("Please annotate points first.")
            elif key == ord('s'):  # 's' for segment
                if self.current_obj_id > 1:
                    self.segmentation_done = True
                    cv2.destroyAllWindows()
                    self.get_logger().info("Annotation complete.  Generating segmentation...")
                    break #  删除 self.generate_segmentation(frame)
                else:
                    self.get_logger().warn("Please add at least one object.")
            elif key == 27:  # ESC key
                rclpy.shutdown()
                self.destroy_node()
                cv2.destroyAllWindows()
                break


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveSegmentationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()