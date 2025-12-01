import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple

class PoseEstimator:
    """Advanced pose estimation for cricket batting analysis"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Cricket-specific pose landmarks indices
        self.cricket_keypoints = {
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13,
            'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15,
            'RIGHT_WRIST': 16,
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_KNEE': 25,
            'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29,
            'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31,
            'RIGHT_FOOT_INDEX': 32
        }
    
    def estimate_pose(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[any]]:
        """Estimate pose from frame and return annotated frame with landmarks"""
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Convert back to BGR
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks with cricket-specific styling
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add cricket-specific annotations
            self._draw_cricket_annotations(annotated_frame, results.pose_landmarks)
        
        return annotated_frame, results.pose_landmarks
    
    def extract_cricket_keypoints(self, landmarks) -> Dict[str, Tuple[float, float, float]]:
        """Extract cricket-specific keypoints as 3D coordinates"""
        
        keypoints = {}
        if landmarks:
            for name, idx in self.cricket_keypoints.items():
                landmark = landmarks.landmark[idx]
                keypoints[name] = (landmark.x, landmark.y, landmark.z)
        
        return keypoints
    
    def _draw_cricket_annotations(self, frame: np.ndarray, landmarks) -> None:
        """Draw cricket-specific pose annotations"""
        
        h, w = frame.shape[:2]
        keypoints = self.extract_cricket_keypoints(landmarks)
        
        if not keypoints:
            return
        
        # Highlight batting arm (right arm for right-handed batsman)
        if all(key in keypoints for key in ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']):
            pts = []
            for key in ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']:
                x, y, _ = keypoints[key]
                pts.append((int(x * w), int(y * h)))
            
            # Draw batting arm chain
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], (0, 255, 0), 3)
        
        # Highlight stance (feet positioning)
        if 'LEFT_ANKLE' in keypoints and 'RIGHT_ANKLE' in keypoints:
            left_x, left_y, _ = keypoints['LEFT_ANKLE']
            right_x, right_y, _ = keypoints['RIGHT_ANKLE']
            
            left_pt = (int(left_x * w), int(left_y * h))
            right_pt = (int(right_x * w), int(right_y * h))
            
            # Draw stance line
            cv2.line(frame, left_pt, right_pt, (255, 0, 0), 2)
            
            # Calculate and display stance width
            stance_width = abs(left_x - right_x) * w
            cv2.putText(frame, f"Stance: {int(stance_width)}px", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_pose_angles(self, landmarks) -> Dict[str, float]:
        """Calculate important angles for cricket batting analysis"""
        
        angles = {}
        keypoints = self.extract_cricket_keypoints(landmarks)
        
        if len(keypoints) < 6:  # Not enough keypoints
            return angles
        
        try:
            # Batting arm angle (shoulder-elbow-wrist)
            if all(key in keypoints for key in ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']):
                shoulder = np.array(keypoints['RIGHT_SHOULDER'][:2])
                elbow = np.array(keypoints['RIGHT_ELBOW'][:2])
                wrist = np.array(keypoints['RIGHT_WRIST'][:2])
                
                angles['batting_arm'] = self._calculate_angle(shoulder, elbow, wrist)
            
            # Front leg angle (hip-knee-ankle)
            if all(key in keypoints for key in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']):
                hip = np.array(keypoints['RIGHT_HIP'][:2])
                knee = np.array(keypoints['RIGHT_KNEE'][:2])
                ankle = np.array(keypoints['RIGHT_ANKLE'][:2])
                
                angles['front_leg'] = self._calculate_angle(hip, knee, ankle)
            
            # Body lean (shoulder line to vertical)
            if 'LEFT_SHOULDER' in keypoints and 'RIGHT_SHOULDER' in keypoints:
                left_shoulder = keypoints['LEFT_SHOULDER']
                right_shoulder = keypoints['RIGHT_SHOULDER']
                
                shoulder_slope = (right_shoulder[1] - left_shoulder[1]) / (right_shoulder[0] - left_shoulder[0] + 1e-6)
                angles['body_lean'] = abs(np.degrees(np.arctan(shoulder_slope)))
        
        except Exception as e:
            print(f"Error calculating angles: {e}")
        
        return angles
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle between three points (in degrees)"""
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
