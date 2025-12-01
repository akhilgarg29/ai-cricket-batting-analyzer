import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pose_analysis.pose_estimator import PoseEstimator
from pose_analysis.cricket_analyzer import CricketBattingAnalyzer, CricketIssue
from pose_analysis.pose_utils import create_pose_summary, draw_angle_annotation

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Install with: pip install ultralytics")


class SmartVideoProcessor:
    """Advanced video processor with integrated pose estimation and cricket analysis"""
    
    def __init__(self, frame_skip: int = 2):
        """
        Initialize the smart video processor
        
        Args:
            frame_skip: Process every Nth frame (default: 2 for better performance)
        """
        print("🔧 Initializing AI Cricket Batting Analyzer...")
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator()
        print("✅ Pose Estimator loaded")
        
        # Initialize cricket analyzer
        self.cricket_analyzer = CricketBattingAnalyzer()
        print("✅ Cricket Analyzer loaded")
        
        # Initialize YOLO for person detection
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                if os.path.exists('yolov8n.pt'):
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("✅ YOLO model loaded")
                else:
                    print("⚠️ yolov8n.pt not found. YOLO detection disabled.")
            except Exception as e:
                print(f"⚠️ YOLO initialization failed: {e}")
        
        self.frame_skip = frame_skip
        self.all_issues = []  # Store all detected issues
        self.frame_analyses = []  # Store per-frame analysis
        
    def process_video(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process cricket batting video with comprehensive analysis
        
        Returns:
            Dictionary containing analysis results, score, and recommendations
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"⚙️ Processing every {self.frame_skip} frame(s) for optimal performance\n")
        
        # Initialize video writer with better codec
        # Initialize video writer with H.264 codec (browser-compatible)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Fallback to mp4v if avc1 fails
        if not out.isOpened():
            print("⚠️ H.264 codec not available, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        
        self.all_issues = []
        self.frame_analyses = []
        frame_count = 0
        processed_count = 0
    
        # ADD THESE VARIABLES
        current_issues = []  # Store current detected issues
        current_angles = {}  # Store current angles
        current_keypoints = {}  # Store current keypoints
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                annotated_frame = frame.copy()
                
                # Process frame at intervals
                if frame_count % self.frame_skip == 0:
                    processed_count += 1
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing: {progress:.1f}% (Frame {frame_count}/{total_frames})")
                    
                    # Perform pose estimation
                    annotated_frame, landmarks = self.pose_estimator.estimate_pose(frame)
                    
                    if landmarks:
                        # Extract keypoints and angles
                        keypoints = self.pose_estimator.extract_cricket_keypoints(landmarks)
                        angles = self.pose_estimator.get_pose_angles(landmarks)
                        
                        # UPDATE CURRENT STATE
                        current_keypoints = keypoints
                        current_angles = angles
                        
                        # Perform comprehensive cricket analysis
                        issues = self.cricket_analyzer.analyze_batting_technique(keypoints, angles)
                        
                        if issues:
                            # UPDATE CURRENT ISSUES
                            current_issues = issues
                            self.all_issues.extend(issues)
                            self.frame_analyses.append({
                                'frame': frame_count,
                                'issues': issues,
                                'angles': angles
                            })
                else:
                    # IMPORTANT: For non-analyzed frames, still draw pose
                    annotated_frame, landmarks = self.pose_estimator.estimate_pose(frame)
                
                # DRAW ANALYSIS ON EVERY FRAME using current state
                if current_issues:
                    self._draw_comprehensive_analysis(
                        annotated_frame, current_issues, current_angles, 
                        current_keypoints, width, height
                    )
                
                # YOLO person detection
                if self.yolo_model and frame_count % (self.frame_skip * 2) == 0:
                    self._draw_person_detection(annotated_frame, frame)
                
                # Add frame info overlay
                self._draw_frame_overlay(annotated_frame, frame_count, width, height)
                
                # Write frame
                out.write(annotated_frame)

        
        finally:
            cap.release()
            out.release()
            print(f"\n✅ Video processing complete! Processed {processed_count} frames")
            print(f"📁 Output saved: {output_path}")
        
        # Generate comprehensive results
        return self._generate_comprehensive_results()
    
    def _draw_comprehensive_analysis(self, frame: np.ndarray, issues: List[CricketIssue],
                                angles: Dict[str, float], keypoints: Dict,
                                width: int, height: int) -> None:
        """Draw comprehensive analysis annotations on frame"""
        
        y_offset = 60
        
        # Draw semi-transparent background for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 40), (min(600, width-5), min(40 + len(issues) * 35 + 60, height-40)), 
                    (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display detected angles - REMOVE EMOJI
        angle_text = " | ".join([f"{k.replace('_', ' ').title()}: {v:.0f}°" 
                                for k, v in angles.items()])
        cv2.putText(frame, f"Angles: {angle_text}", (10, y_offset),  # Changed from "📐 {angle_text}"
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 30
        
        # Display issues
        for issue in issues[:5]:  # Limit to top 5 issues to avoid clutter
            # Color coding by severity
            if issue.severity == 'High':
                color = (0, 0, 255)  # Red
                prefix = "[HIGH]"    # Text prefix instead of emoji
            elif issue.severity == 'Medium':
                color = (0, 165, 255)  # Orange
                prefix = "[MED]"
            elif issue.severity == 'Good':
                color = (0, 255, 0)  # Green
                prefix = "[GOOD]"
            else:
                color = (0, 255, 255)  # Yellow
                prefix = "[LOW]"
            
            text = f"{prefix} {issue.type}: {issue.description[:50]}"  # Removed bullet emoji
            cv2.putText(frame, text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 25
        
        # Draw angle annotations at key joints
        if 'batting_arm' in angles and all(k in keypoints for k in ['RIGHT_ELBOW']):
            elbow = keypoints['RIGHT_ELBOW']
            elbow_px = (int(elbow[0] * width), int(elbow[1] * height))
            draw_angle_annotation(frame, elbow_px, angles['batting_arm'], radius=30)

        def _draw_person_detection(self, annotated_frame: np.ndarray, frame: np.ndarray) -> None:
            """Draw YOLO person detection boxes"""
            try:
                results = self.yolo_model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if int(box.cls) == 0:  # Person class
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f"Batsman {conf:.2f}", 
                                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"⚠️ YOLO processing error: {e}")
    
    def _draw_frame_overlay(self, frame: np.ndarray, frame_count: int, 
                       width: int, height: int) -> None:
        """Draw frame information overlay"""
        # Top-left frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bottom branding - REMOVE EMOJI
        cv2.putText(frame, "AI Cricket Batting Analyzer", (10, height - 20),  # Removed cricket emoji
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    
    def _generate_comprehensive_results(self) -> Dict[str, Any]:
        """Generate comprehensive analysis results with scoring"""
        
        if not self.all_issues:
            return {
                "status": "success",
                "analysis_results": [{
                    "type": "Excellent Form",
                    "description": "Outstanding batting technique! No significant issues detected.",
                    "severity": "Good",
                    "recommendation": "Maintain this excellent form through consistent practice.",
                    "confidence": 0.95
                }],
                "performance_score": 95,
                "performance_grade": "Excellent",
                "total_issues_found": 0,
                "frames_analyzed": len(self.frame_analyses),
                "summary": "✅ Your batting technique demonstrates excellent fundamentals!"
            }
        
        # Aggregate and deduplicate issues
        issue_frequency = {}
        issue_examples = {}
        
        for issue in self.all_issues:
            key = issue.type
            if key not in issue_frequency:
                issue_frequency[key] = 0
                issue_examples[key] = issue
            issue_frequency[key] += 1
        
        # FILTER: Only include issues that appear in >10% of frames
        min_frames = len(self.frame_analyses) * 0.1  # 10% threshold
        filtered_issues = {
            k: v for k, v in issue_examples.items() 
            if issue_frequency[k] >= min_frames
        }
        
        # If too many issues filtered out, it's likely good technique
        if len(filtered_issues) == 0 and len(issue_examples) > 0:
            return {
                "status": "success",
                "analysis_results": [{
                    "type": "Good Form",
                    "description": "Minor inconsistencies detected but overall technique is solid.",
                    "severity": "Good",
                    "recommendation": "Focus on consistency in your movements.",
                    "confidence": 0.85
                }],
                "performance_score": 85,
                "performance_grade": "Excellent",
                "total_issues_found": 1,
                "frames_analyzed": len(self.frame_analyses),
                "summary": "✅ Your batting technique shows strong fundamentals!"
            }
        
        # Use filtered issues for scoring
        unique_issues = list(filtered_issues.values())
        
        # Sort by frequency and severity
        def severity_weight(issue):
            severity_map = {'High': 3, 'Medium': 2, 'Low': 1, 'Good': 0}
            return (severity_map.get(issue.severity, 0), issue_frequency[issue.type])
        
        unique_issues = sorted(unique_issues, key=severity_weight, reverse=True)
        
        # Calculate performance score
        score, grade = self.cricket_analyzer.get_performance_score(unique_issues)
        
        # Create summary
        summary = create_pose_summary(unique_issues[:5])
        
        # Format results for API response
        formatted_issues = []
        for issue in unique_issues[:8]:
            frequency = issue_frequency[issue.type]
            formatted_issues.append({
                "type": issue.type,
                "description": f"{issue.description} (detected in {frequency} frame{'s' if frequency > 1 else ''})",
                "severity": issue.severity,
                "recommendation": issue.recommendation,
                "confidence": issue.confidence,
                "frequency": frequency
            })
        
        return {
            "status": "success",
            "analysis_results": formatted_issues,
            "performance_score": score,
            "performance_grade": grade,
            "total_issues_found": len(unique_issues),
            "frames_analyzed": len(self.frame_analyses),
            "summary": summary
        }

