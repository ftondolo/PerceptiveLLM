#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import torch
import tf2_ros
import tf.transformations
import message_filters
import json
import queue
from cv_bridge import CvBridge
from ultralytics import YOLO
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Header, String
from rtabmap_ros.msg import MapGraph
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

class ObjectDetectionAndNavigation:
    def __init__(self):
        rospy.init_node('object_detection_and_navigation')
        
        # Initialize CV bridge and YOLO model
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        
        # Object tracking dictionary - will store object positions in world frame
        self.tracked_objects = {}  # {id: {'class': class_name, 'position': (x,y,z), 'last_seen': timestamp}}
        
        # TF2 for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize robot state
        self.robot_position = None
        self.current_path = None
        self.occupancy_map = None
        self.map_metadata = None
        
        # Publishers
        self.detection_image_pub = rospy.Publisher('/detection_visualization', Image, queue_size=5)
        self.setpoint_pub = rospy.Publisher('/setpoint', PoseStamped, queue_size=10)
        self.object_markers_pub = rospy.Publisher('/tracked_objects', MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=5)
        
        # Subscribers
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.robot_pose_sub = rospy.Subscriber('/local_position', PoseStamped, self.robot_pose_callback)
        self.occupancy_map_sub = rospy.Subscriber('/rtabmap/grid_map', OccupancyGrid, self.occupancy_map_callback)
        self.map_graph_sub = rospy.Subscriber('/rtabmap/mapGraph', MapGraph, self.map_graph_callback)
        self.pointcloud_sub = rospy.Subscriber('/rtabmap/cloud_map', PointCloud2, self.pointcloud_callback)
        
        # Synchronize color and depth images
        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)
        
        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = None
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        
        rospy.loginfo("Object detection and navigation node initialized")
        
        # Command processing rate
        self.rate = rospy.Rate(10)  # 10 Hz
        
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            rospy.loginfo("Camera calibration parameters received")
    
    def robot_pose_callback(self, msg):
        self.robot_position = msg.pose
        rospy.loginfo(f"Robot position updated: x={self.robot_position.position.x:.2f}, y={self.robot_position.position.y:.2f}, z={self.robot_position.position.z:.2f}")
    
    def occupancy_map_callback(self, msg):
        self.occupancy_map = msg
        self.map_metadata = {
            'width': msg.info.width,
            'height': msg.info.height,
            'resolution': msg.info.resolution,
            'origin_x': msg.info.origin.position.x,
            'origin_y': msg.info.origin.position.y
        }
        rospy.loginfo("Occupancy map updated")
    
    def map_graph_callback(self, msg):
        # Process map topology for navigation planning
        rospy.loginfo(f"Map graph updated: {len(msg.nodes)} nodes, {len(msg.edges)} edges")
    
    def pointcloud_callback(self, msg):
        # Process point cloud data for enhanced environment understanding
        # We'll use this for more accurate depth information and obstacle detection
        rospy.loginfo("Point cloud map updated")
    
    def depth_to_point(self, u, v, depth, frame_id):
        """Convert pixel coordinates and depth to 3D point in camera frame"""
        if self.fx is None:
            rospy.logwarn("Camera calibration not yet received")
            return None
            
        # Convert from pixel to 3D point in camera frame
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        # Create point in camera frame
        point_camera = PoseStamped()
        point_camera.header.frame_id = frame_id
        point_camera.header.stamp = rospy.Time.now()
        point_camera.pose.position.x = z  # x forward in camera frame
        point_camera.pose.position.y = -x  # y left in camera frame
        point_camera.pose.position.z = -y  # z up in camera frame
        point_camera.pose.orientation.w = 1.0
        
        try:
            # Transform to map frame
            point_map = self.tf_buffer.transform(point_camera, "map", rospy.Duration(1.0))
            return point_map.pose.position
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF error: {e}")
            return None
    
    def image_callback(self, color_msg, depth_msg):
        if self.camera_matrix is None:
            rospy.logwarn("Camera calibration not yet received")
            return
            
        # Convert ROS images to OpenCV format
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return
            
        # Normalize depth image for visualization (0-10 meters)
        depth_image_normalized = np.minimum(depth_image, 10000)  # 10m max for visualization
        depth_image_vis = (depth_image_normalized / 10000 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_image_vis, cv2.COLORMAP_JET)
        
        # Run YOLO detection
        results = self.model(color_image)
        
        # Create timestamp for tracking
        current_time = rospy.Time.now()
        
        # Process detections
        detection_image = color_image.copy()
        marker_array = MarkerArray()
        marker_id = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get detection class
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]
                
                # Skip low confidence detections
                if conf < 0.5:
                    continue
                
                # Calculate center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check if center is valid
                if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                    # Get depth at center (convert from mm to meters)
                    center_depth = depth_image[center_y, center_x] / 1000.0
                    
                    # Skip invalid depth
                    if center_depth <= 0 or center_depth > 10:
                        x1, y1, x2, y2 = box
                        rgb_display = color_image.copy()
                        depth_data = depth_image.copy()
                        depth_height, depth_width = depth_data.shape[:2]
                        rgb_height, rgb_width = rgb_display.shape[:2]
                        scale_x, scale_y = (depth_width / rgb_width), (depth_height / rgb_height)
                        x1_depth, y1_depth, x2_depth, y2_depth = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                        
                        # Ensure bounds
                        x1_depth = max(0, min(x1_depth, depth_width - 1))
                        y1_depth = max(0, min(y1_depth, depth_height - 1))
                        x2_depth = max(0, min(x2_depth, depth_width - 1))
                        y2_depth = max(0, min(y2_depth, depth_height - 1))
                        
                        # Sample a small region in the box and take median of non-zero values
                        box_depths = depth_data[y1_depth:y2_depth, x1_depth:x2_depth]
                        non_zero_depths = box_depths[box_depths > 0]
                        
                        if len(non_zero_depths) > 0:
                            center_depth = np.median(non_zero_depths) * self.depth_scale
                        else:
                            continue
                        
                    # Convert to 3D point in map frame
                    map_point = self.depth_to_point(center_x, center_y, center_depth, color_msg.header.frame_id)
                    
                    if map_point:
                        obj_id = f"{cls_name}_{marker_id}"
                        marker_id += 1
                        
                        marker = Marker()
                        marker.header.frame_id = "map"
                        marker.header.stamp = current_time
                        marker.ns = cls_name  # Store class name in the namespace
                        marker.id = marker_id
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD
                        marker.pose.position = map_point
                        marker.pose.orientation.w = 1.0
                        marker.scale.x = 0.2
                        marker.scale.y = 0.2
                        marker.scale.z = 0.2
                        marker.color.r = 1.0 if cls_name == "person" else 0.0
                        marker.color.g = 0.0 if cls_name == "person" else 1.0
                        marker.color.b = 0.0
                        marker.color.a = 0.8
                        marker.lifetime = rospy.Duration(5.0)  # Display for 5 seconds

                        # Add text marker to display class name
                        text_marker = Marker()
                        text_marker.header.frame_id = "map"
                        text_marker.header.stamp = current_time
                        text_marker.ns = "object_labels"
                        text_marker.id = marker_id
                        text_marker.type = Marker.TEXT_VIEW_FACING
                        text_marker.action = Marker.ADD
                        text_marker.pose.position = map_point
                        text_marker.pose.position.z += 0.3  # Position text above the object
                        text_marker.pose.orientation.w = 1.0
                        text_marker.scale.z = 0.1  # Text height
                        text_marker.color.r = 1.0
                        text_marker.color.g = 1.0
                        text_marker.color.b = 1.0
                        text_marker.color.a = 0.8
                        text_marker.text = cls_name
                        text_marker.lifetime = rospy.Duration(5.0)

                        marker_array.markers.append(marker)
                        marker_array.markers.append(text_marker)

                        # Also update the tracked_objects dictionary to include more information
                        self.tracked_objects[obj_id] = {
                            'class': cls_name,
                            'position': (map_point.x, map_point.y, map_point.z),
                            'last_seen': current_time,
                            'confidence': conf  # Store the confidence score from YOLO
                        }

                        # Draw on image
                        label = f"{cls_name}: {conf:.2f}, {center_depth:.2f}m"
                        cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(detection_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(detection_image, (center_x, center_y), 5, (0, 0, 255), -1)
                        
                        rospy.loginfo(f"Detected {cls_name} at distance {center_depth:.2f}m, map position: ({map_point.x:.2f}, {map_point.y:.2f}, {map_point.z:.2f})")
        
        # Publish detection visualization
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(detection_image, "bgr8")
            detection_msg.header = color_msg.header
            self.detection_image_pub.publish(detection_msg)
            
            # Publish object markers
            if marker_array.markers:
                self.object_markers_pub.publish(marker_array)
        except Exception as e:
            rospy.logerr(f"Error publishing detection: {e}")
    
    def prune_stale_objects(self):
        """Remove objects that haven't been seen recently"""
        current_time = rospy.Time.now()
        stale_threshold = rospy.Duration(30.0)  # 30 seconds
        
        stale_objects = []
        for obj_id, obj_data in self.tracked_objects.items():
            if current_time - obj_data['last_seen'] > stale_threshold:
                stale_objects.append(obj_id)
        
        for obj_id in stale_objects:
            del self.tracked_objects[obj_id]
    
    def plan_path(self, start_point, goal_point):
        """Plan a path using A* algorithm on the occupancy grid"""
        if self.occupancy_map is None:
            rospy.logwarn("No occupancy map available for path planning")
            return None
        
        # Convert world coordinates to grid coordinates
        def world_to_grid(wx, wy):
            gx = int((wx - self.map_metadata['origin_x']) / self.map_metadata['resolution'])
            gy = int((wy - self.map_metadata['origin_y']) / self.map_metadata['resolution'])
            return gx, gy
        
        def grid_to_world(gx, gy):
            wx = gx * self.map_metadata['resolution'] + self.map_metadata['origin_x']
            wy = gy * self.map_metadata['resolution'] + self.map_metadata['origin_y']
            return wx, wy
        
        def is_valid(x, y):
            if x < 0 or y < 0 or x >= self.map_metadata['width'] or y >= self.map_metadata['height']:
                return False
            
            # Check if obstacle (occupied = 100, free = 0, unknown = -1)
            index = y * self.map_metadata['width'] + x
            return index < len(self.occupancy_map.data) and self.occupancy_map.data[index] < 50
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        start_grid = world_to_grid(start_point.x, start_point.y)
        goal_grid = world_to_grid(goal_point.x, goal_point.y)
        
        # A* algorithm
        open_set = {start_grid}
        closed_set = set()
        
        came_from = {}
        
        g_score = {start_grid: 0}
        f_score = {start_grid: heuristic(start_grid, goal_grid)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    world_point = grid_to_world(current[0], current[1])
                    path.append(world_point)
                    current = came_from[current]
                
                path.reverse()
                
                # Add the goal point
                path.append((goal_point.x, goal_point.y))
                
                # Create ROS Path message
                path_msg = Path()
                path_msg.header.frame_id = "map"
                path_msg.header.stamp = rospy.Time.now()
                
                for wp in path:
                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x = wp[0]
                    pose.pose.position.y = wp[1]
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    path_msg.poses.append(pose)
                
                return path_msg
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Check neighbors (8-connected grid)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in closed_set or not is_valid(neighbor[0], neighbor[1]):
                    continue
                
                # Movement cost (diagonal movement costs more)
                move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                
                tentative_g_score = g_score.get(current, float('inf')) + move_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_grid)
        
        rospy.logwarn("No path found to goal")
        return None
    
    def move_to_setpoint(self, x, y, z):
        """Send a setpoint command to move the robot"""
        if self.robot_position is None:
            rospy.logwarn("Robot position unknown, cannot plan path")
            return False
        
        # Create goal point
        goal = Point()
        goal.x = x
        goal.y = y
        goal.z = z
        
        # Plan path
        path = self.plan_path(self.robot_position.position, goal)
        
        if path:
            # Publish path for visualization
            self.path_pub.publish(path)
            
            # Send setpoint command (using the first waypoint in the path)
            if len(path.poses) > 1:  # First waypoint is current position
                next_waypoint = path.poses[1]
                
                setpoint = PoseStamped()
                setpoint.header.frame_id = "map"
                setpoint.header.stamp = rospy.Time.now()
                setpoint.pose = next_waypoint.pose
                
                self.setpoint_pub.publish(setpoint)
                rospy.loginfo(f"Moving to setpoint: ({next_waypoint.pose.position.x:.2f}, {next_waypoint.pose.position.y:.2f}, {next_waypoint.pose.position.z:.2f})")
                
                # Save current path
                self.current_path = path
                return True
        
        return False
    
    def navigate_to_object(self, obj_id, distance=1.0):
        """Navigate to a specific object with the specified standoff distance"""
        if obj_id not in self.tracked_objects:
            rospy.logwarn(f"Object {obj_id} not found in tracked objects")
            return False
        
        obj_data = self.tracked_objects[obj_id]
        obj_pos = obj_data['position']
        
        # Calculate approach vector 
        if self.robot_position is None:
            rospy.logwarn("Robot position unknown, cannot plan approach")
            return False
        
        # Vector from object to robot
        dx = self.robot_position.position.x - obj_pos[0]
        dy = self.robot_position.position.y - obj_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Normalize and scale to standoff distance
        if dist > 0.001:
            dx = dx / dist * distance
            dy = dy / dist * distance
        else:
            dx, dy = distance, 0  # Default approach direction if very close
        
        # Goal position at standoff distance
        goal_x = obj_pos[0] + dx
        goal_y = obj_pos[1] + dy
        goal_z = 0.0  # Assume flat ground
        
        return self.move_to_setpoint(goal_x, goal_y, goal_z)
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting object detection and navigation control loop")
        
        # Subscribe to LLM commands topic
        self.llm_command_sub = rospy.Subscriber('/llm_commands', String, self.process_llm_command)
        
        # Setup command queue for thread-safe handling
        self.command_queue = queue.Queue()
        
        # Setup command result publisher
        self.command_result_pub = rospy.Publisher('/command_results', String, queue_size=10)
        
        while not rospy.is_shutdown():
            # Prune stale objects
            self.prune_stale_objects()
            
            # Process any commands in the queue
            try:
                # Non-blocking check for new commands
                if not self.command_queue.empty():
                    command = self.command_queue.get(block=False)
                    result = self.execute_command(command)
                    
                    # Publish result
                    result_msg = String()
                    result_msg.data = json.dumps(result)
                    self.command_result_pub.publish(result_msg)
            except queue.Empty:
                pass  # No commands in queue
            except Exception as e:
                rospy.logerr(f"Error processing command: {e}")
            
            # Sleep to maintain loop rate
            self.rate.sleep()
    
    def process_llm_command(self, msg):
        """Process commands from the LLM agent"""
        try:
            # Parse JSON command
            command = json.loads(msg.data)
            
            # Add to command queue
            self.command_queue.put(command)
            
            rospy.loginfo(f"Received command: {command}")
        except json.JSONDecodeError:
            rospy.logerr(f"Received invalid JSON command: {msg.data}")
        except Exception as e:
            rospy.logerr(f"Error processing command: {e}")
    
    def execute_command(self, command):
        """Execute a command from the LLM agent"""
        command_type = command.get('type', '').upper()
        
        if command_type == 'MOVE_TO_COORDINATES':
            # Extract coordinates
            x = command.get('x', 0.0)
            y = command.get('y', 0.0)
            z = command.get('z', 0.0)
            
            # Move to coordinates
            success = self.move_to_setpoint(x, y, z)
            
            return {
                'status': 'success' if success else 'failed',
                'command_type': command_type,
                'message': f"Moving to coordinates ({x}, {y}, {z})" if success else "Failed to plan path to coordinates"
            }
        
        elif command_type == 'MOVE_TO_OBJECT':
            # Extract object ID and standoff distance
            obj_id = command.get('object_id', '')
            standoff_distance = command.get('standoff_distance', 1.0)
            
            # Navigate to object
            success = self.navigate_to_object(obj_id, standoff_distance)
            
            return {
                'status': 'success' if success else 'failed',
                'command_type': command_type,
                'message': f"Moving to object {obj_id}" if success else f"Failed to find or navigate to object {obj_id}"
            }
        
        elif command_type == 'EXPLORE':
            # Extract exploration parameters
            max_distance = command.get('max_distance', 3.0)
            
            # Generate random exploration target
            if self.robot_position is None:
                return {
                    'status': 'failed',
                    'command_type': command_type,
                    'message': "Robot position unknown, cannot plan exploration"
                }
            
            # Generate random angle and use desired max distance
            import random
            angle = random.uniform(0, 2 * np.pi)
            
            # Calculate target position
            x = self.robot_position.position.x + max_distance * np.cos(angle)
            y = self.robot_position.position.y + max_distance * np.sin(angle)
            
            # Check if target is valid using occupancy map
            if self.occupancy_map is not None:
                # Convert world coordinates to grid coordinates
                gx = int((x - self.map_metadata['origin_x']) / self.map_metadata['resolution'])
                gy = int((y - self.map_metadata['origin_y']) / self.map_metadata['resolution'])
                
                # Check if valid point in map
                if (0 <= gx < self.map_metadata['width'] and 
                    0 <= gy < self.map_metadata['height']):
                    
                    # Check if free space (0 = free, 100 = occupied, -1 = unknown)
                    index = gy * self.map_metadata['width'] + gx
                    if index < len(self.occupancy_map.data) and self.occupancy_map.data[index] >= 50:
                        # Target is occupied or unknown, try another point
                        return {
                            'status': 'failed',
                            'command_type': command_type,
                            'message': "Exploration target is in occupied space, try again"
                        }
            
            # Move to exploration target
            success = self.move_to_setpoint(x, y, 0.0)
            
            return {
                'status': 'success' if success else 'failed',
                'command_type': command_type,
                'message': f"Exploring: moving to ({x}, {y})" if success else "Failed to plan exploration path"
            }
        
        elif command_type == 'GET_ENVIRONMENT_DATA':
            # Return data about the environment and detected objects
            env_data = {
                'robot_position': {
                    'x': self.robot_position.position.x if self.robot_position else None,
                    'y': self.robot_position.position.y if self.robot_position else None,
                    'z': self.robot_position.position.z if self.robot_position else None
                },
                'tracked_objects': {
                    obj_id: {
                        'class': obj_data.get('class', 'unknown'),
                        'position': obj_data['position'],
                        'last_seen': obj_data['last_seen'].to_sec()
                    } for obj_id, obj_data in self.tracked_objects.items()
                }
            }
            
            return {
                'status': 'success',
                'command_type': command_type,
                'data': env_data
            }
        
        elif command_type == 'STOP':
            # Stop any current movement
            # For simplicity, we'll just send a setpoint at the current position
            if self.robot_position:
                self.setpoint_pub.publish(PoseStamped(
                    header=Header(frame_id="map", stamp=rospy.Time.now()),
                    pose=self.robot_position
                ))
                
                return {
                    'status': 'success',
                    'command_type': command_type,
                    'message': "Robot stopped"
                }
            else:
                return {
                    'status': 'failed',
                    'command_type': command_type,
                    'message': "Robot position unknown, cannot stop"
                }
        
        elif command_type == 'SCAN_AREA':
            # Perform a 360-degree scan of the current area
            # This involves rotating in place to capture objects in all directions
            if self.robot_position is None:
                return {
                    'status': 'failed',
                    'command_type': command_type,
                    'message': "Robot position unknown, cannot perform scan"
                }
            
            # Get current orientation as quaternion
            current_orientation = self.robot_position.orientation
            
            # Convert to Euler angles
            euler = tf.transformations.euler_from_quaternion([
                current_orientation.x,
                current_orientation.y,
                current_orientation.z,
                current_orientation.w
            ])
            
            # Current yaw angle
            current_yaw = euler[2]
            
            # Number of scan points (e.g., 8 points for a 360-degree scan)
            num_points = 8
            angle_increment = 2 * np.pi / num_points
            
            # Create setpoints for each scan angle
            scan_setpoints = []
            for i in range(num_points):
                # Calculate new yaw angle
                new_yaw = current_yaw + i * angle_increment
                
                # Convert back to quaternion
                new_quaternion = tf.transformations.quaternion_from_euler(0, 0, new_yaw)
                
                # Create setpoint at current position but with new orientation
                setpoint = PoseStamped()
                setpoint.header.frame_id = "map"
                setpoint.header.stamp = rospy.Time.now()
                setpoint.pose.position = self.robot_position.position
                setpoint.pose.orientation.x = new_quaternion[0]
                setpoint.pose.orientation.y = new_quaternion[1]
                setpoint.pose.orientation.z = new_quaternion[2]
                setpoint.pose.orientation.w = new_quaternion[3]
                
                scan_setpoints.append(setpoint)
            
            # Execute scan by publishing setpoints one by one with delays
            for i, setpoint in enumerate(scan_setpoints):
                self.setpoint_pub.publish(setpoint)
                rospy.sleep(1.0)  # Wait for robot to rotate
                
                # Update scan progress
                progress = {
                    'status': 'in_progress',
                    'command_type': command_type,
                    'message': f"Scanning: {i+1}/{num_points} complete"
                }
                result_msg = String()
                result_msg.data = json.dumps(progress)
                self.command_result_pub.publish(result_msg)
            
            return {
                'status': 'success',
                'command_type': command_type,
                'message': f"Area scan complete, {len(self.tracked_objects)} objects detected"
            }
        
        else:
            # Unknown command
            return {
                'status': 'failed',
                'command_type': 'unknown',
                'message': f"Unknown command type: {command_type}"
            }

if __name__ == '__main__':
    try:
        node = ObjectDetectionAndNavigation()
        node.run()
    except rospy.ROSInterruptException:
        pass
