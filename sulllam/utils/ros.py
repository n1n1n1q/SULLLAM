import rclpy
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

class ROSPublisherWrapper:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('my_external_app_node')
        
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        self.path_pubisher = self.node.create_publisher(Path, 'path', 10)

        self.image_pair_publisher = self.node.create_publisher(Image, 'current_pair', 10)
        self.matches_publisher = self.node.create_publisher(Image, 'curent_matches', 10)
        
        self.bridge = CvBridge()
        
    def publish_pose(self, translation: np.ndarray, orientation: np.ndarray, frame_id="map"):
        """
        Publishes a PoseStamped message.
        Defaults orientation to no rotation (qw=1.0) and frame to 'map'.
        """
        msg = PoseStamped()
        
        x, y, z = translation
        qx, qy, qz, qw = orientation

        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        
        self.path_msg.poses.append(msg)

        self.path_pubisher.publish(self.path_msg)

    def publish_current_pair(self, cv_image, frame_id="camera_link"):
        msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        self.image_pair_publisher.publish(msg)

    def publish_current_matches(self, cv_image, frame_id="camera_link"):
        msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        self.matches_publisher.publish(msg)

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()
        print("[ROS] Node shut down.")