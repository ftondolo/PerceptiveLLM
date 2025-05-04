#!/usr/bin/env python3

import rospy
import json
import numpy as np
import requests
import time
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class OllamaRobotAgent:
    def __init__(self):
        rospy.init_node('ollama_robot_agent')
        
        # Initialize state
        self.bridge = CvBridge()
        self.robot_position = None
        self.tracked_objects = {}
        self.current_image = None
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3"  # Adjust based on your Ollama setup
        
        # Track command results
        self.last_command_result = None
        self.command_history = []  # Store recent commands and results
        
        # System prompt for the LLM
        self.system_prompt = """
        You are an AI agent controlling a robot. The robot has a camera that can detect objects and measure distances.
        Your goal is to explore the environment and interact with objects of interest.
        
        You receive:
        1. The robot's current position
        2. A list of detected objects with their positions and distances
        3. A description of what the robot camera sees
        
        You can issue the following commands:
        - MOVE_TO_COORDINATES(x, y): Move to specific coordinates
        - MOVE_TO_OBJECT(object_id): Move near a specific detected object
        - EXPLORE(): Move to unexplored areas
        - GET_ENVIRONMENT_DATA(): Get detailed data about the environment
        - STOP(): Stop robot movement immediately
        - SCAN_AREA(): Perform a 360-degree scan of the surroundings
        
        Always think carefully about what action would be most useful given the current state.
        """
        
        # Subscribers
        self.robot_pose_sub = rospy.Subscriber('/local_position', PoseStamped, self.robot_pose_callback)
        self.objects_sub = rospy.Subscriber('/tracked_objects', MarkerArray, self.objects_callback)
        self.image_sub = rospy.Subscriber('/detection_visualization', Image, self.image_callback)
        self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
        
        # Publishers
        self.command_pub = rospy.Publisher('/llm_commands', String, queue_size=10)
        
        # Subscribe to command results
        self.results_sub = rospy.Subscriber('/command_results', String, self.command_result_callback)
        
        # Control rate
        self.rate = rospy.Rate(1)  # 1 Hz decision making
        
        # Path tracking
        self.current_path = None
        self.path_index = 0
        
        # Memory for object permanence
        self.object_memory = {}  # {id: {'class': class_name, 'position': (x,y,z), 'last_seen': timestamp, 'confidence': value}}
        
        rospy.loginfo("Ollama robot agent initialized")
    
    def robot_pose_callback(self, msg):
        self.robot_position = msg.pose
    
    def objects_callback(self, msg):
        """Update tracked objects from markers"""
        current_time = rospy.Time.now()
        
        for marker in msg.markers:
            # Skip text markers
            if marker.type == Marker.TEXT_VIEW_FACING:
                continue
                
            obj_id = f"object_{marker.id}"
            position = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            
            # Extract class from marker namespace
            # If namespace is empty, fall back to color-based classification
            if marker.ns and marker.ns != "objects":
                obj_class = marker.ns
            else:
                # Fallback to color-based classification
                color = (marker.color.r, marker.color.g, marker.color.b)
                if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                    obj_class = "person"
                else:
                    obj_class = "object"
            
            # Update currently tracked objects
            self.tracked_objects[obj_id] = {
                'position': position,
                'color': (marker.color.r, marker.color.g, marker.color.b),
                'class': obj_class
            }
            
            # Update object memory for permanence
            if obj_id in self.object_memory:
                # Object already known - update its position and increase confidence
                self.object_memory[obj_id]['position'] = position
                self.object_memory[obj_id]['last_seen'] = current_time
                self.object_memory[obj_id]['confidence'] = min(1.0, self.object_memory[obj_id]['confidence'] + 0.1)
                self.object_memory[obj_id]['class'] = obj_class  # Update class in case it changed
            else:
                # New object - add to memory
                self.object_memory[obj_id] = {
                    'position': position,
                    'class': obj_class,
                    'last_seen': current_time,
                    'confidence': 0.6  # Initial confidence
                }
                
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
    
    def path_callback(self, msg):
        self.current_path = msg
        self.path_index = 0
    
    def update_object_memory(self):
        """Update object memory to implement object permanence"""
        current_time = rospy.Time.now()
        memory_timeout = rospy.Duration(300.0)  # 5 minutes
        confidence_decay = 0.01  # Decay rate for confidence when object not seen
        
        # Find objects to decay or remove
        objects_to_remove = []
        
        for obj_id, obj_data in self.object_memory.items():
            time_since_last_seen = current_time - obj_data['last_seen']
            
            # If object is not currently tracked, decay its confidence
            if obj_id not in self.tracked_objects:
                self.object_memory[obj_id]['confidence'] -= confidence_decay
                
                # If confidence too low or timeout exceeded, mark for removal
                if (self.object_memory[obj_id]['confidence'] <= 0.1 or 
                    time_since_last_seen > memory_timeout):
                    objects_to_remove.append(obj_id)
        
        # Remove objects with low confidence
        for obj_id in objects_to_remove:
            del self.object_memory[obj_id]
    
    def generate_state_description(self):
        """Generate a text description of the current state for the LLM"""
        if self.robot_position is None:
            return "No robot position data available yet."
        
        description = f"Robot position: x={self.robot_position.position.x:.2f}, y={self.robot_position.position.y:.2f}, z={self.robot_position.position.z:.2f}\n\n"
        
        # Currently visible objects
        description += "Currently visible objects:\n"
        if not self.tracked_objects:
            description += "No objects currently in view.\n"
        else:
            for obj_id, obj_data in self.tracked_objects.items():
                pos = obj_data['position']
                obj_class = obj_data.get('class', 'unknown')
                
                # Calculate distance from robot to object
                dx = pos[0] - self.robot_position.position.x
                dy = pos[1] - self.robot_position.position.y
                dz = pos[2] - self.robot_position.position.z
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                description += f"- {obj_id} (class: {obj_class}): position=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), distance={distance:.2f}m\n"
        
        # Objects in memory (not currently visible)
        description += "\nObjects in memory (not currently visible):\n"
        memory_only_objects = [obj_id for obj_id in self.object_memory if obj_id not in self.tracked_objects]
        
        if not memory_only_objects:
            description += "No additional objects in memory.\n"
        else:
            for obj_id in memory_only_objects:
                obj_data = self.object_memory[obj_id]
                pos = obj_data['position']
                confidence = obj_data['confidence']
                obj_class = obj_data.get('class', 'unknown')
                
                # Calculate time since last seen
                time_since_seen = (rospy.Time.now() - obj_data['last_seen']).to_sec()
                
                # Calculate distance from robot to object
                dx = pos[0] - self.robot_position.position.x
                dy = pos[1] - self.robot_position.position.y
                dz = pos[2] - self.robot_position.position.z
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                description += f"- {obj_id} (class: {obj_class}): position=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), distance={distance:.2f}m, last seen {time_since_seen:.1f}s ago, confidence={confidence:.2f}\n"
        
        # Add information about current path if available
        if self.current_path:
            description += f"\nCurrently following a path with {len(self.current_path.poses)} waypoints.\n"
            description += f"At waypoint {self.path_index} of {len(self.current_path.poses)}.\n"
        
        return description
    
    def query_llm(self, state_description):
        """Send the current state to the LLM and get a command"""
        prompt = f"{self.system_prompt}\n\nCurrent state:\n{state_description}\n\nWhat should the robot do next?"
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                rospy.logerr(f"Error from Ollama API: {response.status_code} - {response.text}")
                return "ERROR: Could not get response from language model."
        
        except Exception as e:
            rospy.logerr(f"Exception when calling Ollama API: {e}")
            return "ERROR: Exception when communicating with language model."
    
    def command_result_callback(self, msg):
        """Process command results from the navigation node"""
        try:
            result = json.loads(msg.data)
            status = result.get('status')
            command_type = result.get('command_type')
            message = result.get('message', '')
            
            rospy.loginfo(f"Command result: {status} - {message}")
            
            # Store result for potential reference by the LLM
            self.last_command_result = result
            
        except json.JSONDecodeError:
            rospy.logerr(f"Received invalid JSON result: {msg.data}")
        except Exception as e:
            rospy.logerr(f"Error processing command result: {e}")
    
    def parse_command(self, llm_response):
        """Parse the LLM's response into actionable commands"""
        # Look for command patterns in the response
        response_lower = llm_response.lower()
        
        if "move_to_coordinates(" in response_lower:
            # Extract coordinates
            try:
                # Find the command and extract parameters
                start_idx = response_lower.find("move_to_coordinates(") + 19
                end_idx = response_lower.find(")", start_idx)
                if end_idx > start_idx:
                    coords = response_lower[start_idx:end_idx].split(',')
                    if len(coords) >= 2:
                        x = float(coords[0].strip())
                        y = float(coords[1].strip())
                        z = 0.0  # Default Z coordinate
                        
                        # Create command JSON
                        command = {
                            'type': 'MOVE_TO_COORDINATES',
                            'x': x,
                            'y': y,
                            'z': z
                        }
                        
                        # Send the command
                        self.send_command(command)
                        return f"Sending MOVE_TO_COORDINATES command: ({x}, {y})"
            except Exception as e:
                rospy.logerr(f"Error parsing MOVE_TO command: {e}")
        
        elif "move_to_object(" in response_lower:
            try:
                # Find the command and extract parameters
                start_idx = response_lower.find("move_to_object(") + 15
                end_idx = response_lower.find(")", start_idx)
                if end_idx > start_idx:
                    obj_id = response_lower[start_idx:end_idx].strip()
                    
                    # Remove quotes if present
                    if obj_id.startswith('"') and obj_id.endswith('"'):
                        obj_id = obj_id[1:-1]
                    
                    # Create command JSON
                    command = {
                        'type': 'MOVE_TO_OBJECT',
                        'object_id': obj_id,
                        'standoff_distance': 1.0
                    }
                    
                    # Send the command
                    self.send_command(command)
                    return f"Sending MOVE_TO_OBJECT command for object: {obj_id}"
            except Exception as e:
                rospy.logerr(f"Error parsing MOVE_TO_OBJECT command: {e}")
        
        elif "explore(" in response_lower or "explore()" in response_lower:
            # Extract max_distance if provided
            max_distance = 3.0  # Default
            if "explore(" in response_lower:
                try:
                    start_idx = response_lower.find("explore(") + 8
                    end_idx = response_lower.find(")", start_idx)
                    if end_idx > start_idx:
                        param_str = response_lower[start_idx:end_idx].strip()
                        if param_str:
                            max_distance = float(param_str)
                except:
                    pass  # Use default if parsing fails
            
            # Create command JSON
            command = {
                'type': 'EXPLORE',
                'max_distance': max_distance
            }
            
            # Send the command
            self.send_command(command)
            return f"Sending EXPLORE command with max_distance: {max_distance}"
        
        elif "scan_area(" in response_lower or "scan_area()" in response_lower:
            # Create command JSON for 360-degree scan
            command = {
                'type': 'SCAN_AREA'
            }
            
            # Send the command
            self.send_command(command)
            return "Sending SCAN_AREA command to perform a 360-degree scan"
        
        elif "stop(" in response_lower or "stop()" in response_lower:
            # Create command JSON to stop the robot
            command = {
                'type': 'STOP'
            }
            
            # Send the command
            self.send_command(command)
            return "Sending STOP command to halt robot movement"
            
        # Additional helper for people who might try to use old MOVE_TO command
        elif "move_to(" in response_lower and "move_to_coordinates(" not in response_lower and "move_to_object(" not in response_lower:
            try:
                # Find the command and extract parameters
                start_idx = response_lower.find("move_to(") + 8
                end_idx = response_lower.find(")", start_idx)
                if end_idx > start_idx:
                    coords = response_lower[start_idx:end_idx].split(',')
                    if len(coords) >= 2:
                        x = float(coords[0].strip())
                        y = float(coords[1].strip())
                        z = 0.0  # Default Z coordinate
                        
                        # Create command JSON
                        command = {
                            'type': 'MOVE_TO_COORDINATES',
                            'x': x,
                            'y': y,
                            'z': z
                        }
                        
                        # Send the command
                        self.send_command(command)
                        return f"Sending MOVE_TO_COORDINATES command: ({x}, {y}) (converted from MOVE_TO)"
            except Exception as e:
                rospy.logerr(f"Error parsing MOVE_TO command: {e}")
        
        elif "get_environment_data(" in response_lower or "get_environment_data()" in response_lower:
            # Create command to get full environment data
            command = {
                'type': 'GET_ENVIRONMENT_DATA'
            }
            
            # Send the command
            self.send_command(command)
            
            # Return message about the command
            return "Sending GET_ENVIRONMENT_DATA command to get detailed environment information"
        
        # If no command matched, just log the response
        rospy.loginfo(f"LLM response (no command detected): {llm_response}")
        return "No actionable command detected"
    
    def send_command(self, command):
        """Send a command to the navigation node"""
        command_msg = String()
        command_msg.data = json.dumps(command)
        
        self.command_pub.publish(command_msg)
        rospy.loginfo(f"Sent command: {command}")
    
    def generate_command_history(self):
        """Generate a string representing recent command history"""
        if not self.command_history:
            return "No previous commands."
        
        history_str = "Recent command history:\n"
        for idx, entry in enumerate(self.command_history[-5:]):  # Last 5 commands
            history_str += f"{idx+1}. Command: {entry['command']}\n   Result: {entry['result']}\n"
        
        return history_str
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting Ollama robot agent control loop")
        
        while not rospy.is_shutdown():
            # Update object memory
            self.update_object_memory()
            
            # Generate state description
            state_description = self.generate_state_description()
            
            # Generate command history
            command_history = self.generate_command_history()
            
            # Create full prompt with state and command history
            full_prompt = f"{state_description}\n\n{command_history}"
            
            # Query the LLM for next action
            llm_response = self.query_llm(full_prompt)
            
            # Parse and execute command
            command_result = self.parse_command(llm_response)
            rospy.loginfo(f"Command interpreted: {command_result}")
            
            # Add to command history
            self.command_history.append({
                'command': command_result,
                'result': self.last_command_result if self.last_command_result else "Pending"
            })
            
            # Keep command history to reasonable size
            if len(self.command_history) > 20:
                self.command_history = self.command_history[-20:]
            
            # Sleep to maintain loop rate
            self.rate.sleep()

if __name__ == '__main__':
    try:
        agent = OllamaRobotAgent()
        agent.run()
    except rospy.ROSInterruptException:
        pass
