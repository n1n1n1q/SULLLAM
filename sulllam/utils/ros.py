import rclpy
import cv2 as cv
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2, PointField

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

        self.pointcloud_publisher = self.node.create_publisher(PointCloud2, 'map_points', 10)
        self.segmentation_publisher = self.node.create_publisher(Image, 'segmentation_overlay', 10)

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
    
    def publish_trajectory(self, translations, orientations, frame_id="map"):
        """
        Clears the existing path and republishes the entire trajectory.
        Use this after Local Bundle Adjustment has modified historical poses.
        
        :param translations: List or Nx3 numpy array of translations (x, y, z).
        :param orientations: List or Nx4 numpy array of quaternions (qx, qy, qz, qw).
        :param frame_id: The reference frame for the path.
        """
        if len(translations) != len(orientations):
            self.node.get_logger().warn("Translations and orientations length mismatch. Cannot publish trajectory.")
            return

        # Clear the existing unoptimized path
        self.path_msg.poses.clear()
        
        # Update the main path header
        current_time = self.node.get_clock().now().to_msg()
        self.path_msg.header.stamp = current_time
        self.path_msg.header.frame_id = frame_id

        # Rebuild the path with the optimized poses
        for trans, rot in zip(translations, orientations):
            msg = PoseStamped()
            msg.header.stamp = current_time
            msg.header.frame_id = frame_id
            
            msg.pose.position.x = float(trans[0])
            msg.pose.position.y = float(trans[1])
            msg.pose.position.z = float(trans[2])
            
            msg.pose.orientation.x = float(rot[0])
            msg.pose.orientation.y = float(rot[1])
            msg.pose.orientation.z = float(rot[2])
            msg.pose.orientation.w = float(rot[3])
            
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

    def publish_pointcloud(self, points_3d: np.ndarray, colors: np.ndarray, frame_id="map"):
        """
        Efficiently packs NumPy arrays into a sensor_msgs/PointCloud2 message and publishes it.
        """
        if len(points_3d) == 0:
            return

        msg = PointCloud2()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        msg.height = 1
        msg.width = len(points_3d)

        # RViz expects a single 'rgb' field packed as a 32-bit integer
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16  # 16 bytes per point (4 bytes * 3 for xyz + 4 bytes for packed rgb)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        # Pack data into an aligned structured numpy array
        cloud_data = np.zeros(len(points_3d), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])

        cloud_data['x'] = points_3d[:, 0]
        cloud_data['y'] = points_3d[:, 1]
        cloud_data['z'] = points_3d[:, 2]

        # Convert colors to uint32 to prepare for bit shifting
        r = np.asarray(colors[:, 0], dtype=np.uint32)
        g = np.asarray(colors[:, 1], dtype=np.uint32)
        b = np.asarray(colors[:, 2], dtype=np.uint32)

        # Pack R, G, B into a single 32-bit integer (0x00RRGGBB)
        cloud_data['rgb'] = (r << 16) | (g << 8) | b

        # Convert straight to bytes
        msg.data = cloud_data.tobytes()
        
        self.pointcloud_publisher.publish(msg)

    def publish_segmentation_overlay(self, cv_image, mask, frame_id="camera_link"):
        """Publish original image and segmentation mask overlay side by side.

        Left half: original frame.
        Right half: frame with the exclusion mask blended in red.
        Topic: /segmentation_overlay
        """
        import numpy as np

        overlay = cv_image.copy()
        if mask is not None and mask.any():
            colored = np.zeros_like(cv_image)
            colored[mask] = (0, 0, 200)  # red in BGR
            overlay = cv.addWeighted(cv_image, 1.0, colored, 0.45, 0)

        side_by_side = cv.hconcat([cv_image, overlay])
        msg = self.bridge.cv2_to_imgmsg(side_by_side, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        self.segmentation_publisher.publish(msg)

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()
        print("[ROS] Node shut down.")