"""
Utility functions for pose analysis
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict

def normalize_landmarks(landmarks, frame_width: int, frame_height: int) -> Dict:
    """Normalize landmarks to frame dimensions"""
    normalized = {}
    
    for name, (x, y, z) in landmarks.items():
        normalized[name] = (
            x * frame_width,
            y * frame_height,
            z
        )
    
    return normalized

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_angle_annotation(frame: np.ndarray, center: Tuple[int, int], 
                         angle: float, radius: int = 50) -> None:
    """Draw angle annotation on frame"""
    
    # Draw arc to show angle
    start_angle = -30
    end_angle = int(angle) - 30
    
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 2)
    
    # Add angle text
    text_pos = (center[0] + radius + 10, center[1])
    cv2.putText(frame, f"{int(angle)}°", text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def create_pose_summary(issues: List) -> str:
    """Create a text summary of pose analysis results"""
    
    if not issues:
        return "✅ Excellent batting technique! No major issues detected."
    
    summary = f"📊 Analysis found {len(issues)} areas for improvement:\n\n"
    
    for i, issue in enumerate(issues, 1):
        severity_emoji = "🔴" if issue.severity == "High" else "🟡" if issue.severity == "Medium" else "🟢"
        summary += f"{severity_emoji} {i}. **{issue.type}**\n"
        summary += f"   {issue.description}\n"
        summary += f"   💡 *{issue.recommendation}*\n\n"
    
    return summary

def filter_confident_landmarks(keypoints: Dict, min_confidence: float = 0.5) -> Dict:
    """Filter out landmarks with low confidence"""
    
    # This is a placeholder - MediaPipe landmarks don't have confidence scores
    # In practice, you'd implement visibility-based filtering
    return {k: v for k, v in keypoints.items() if len(v) >= 3 and v[2] > -0.5}  # z-coordinate as proxy
