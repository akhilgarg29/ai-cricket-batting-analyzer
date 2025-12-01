import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CricketIssue:
    """Data class for cricket batting issues"""
    type: str
    description: str
    severity: str  # 'High', 'Medium', 'Low'
    recommendation: str
    confidence: float

class CricketBattingAnalyzer:
    """Advanced cricket batting technique analyzer"""
    
    def __init__(self):
        # Optimal ranges for cricket batting (based on coaching standards)
        self.optimal_ranges = {
        'batting_arm_angle': (80, 160),    # Was (90, 150) - wider range
        'front_leg_angle': (150, 180),     # Was (160, 180) - allow more bend
        'body_lean': (0, 20),              # Was (0, 15) - allow more lean
        'stance_width': (0.25, 0.7),       # Was (0.3, 0.6) - wider stance ok
        'shoulder_alignment': (0, 15),     # Was (0, 10) - more tolerance
        'hip_alignment': (0, 12)           # Was (0, 8) - more tolerance
        }
        
        # Minimum confidence threshold for analysis
        self.min_confidence = 0.5
    
    def analyze_batting_technique(self, keypoints: Dict, angles: Dict) -> List[CricketIssue]:
        """Comprehensive cricket batting technique analysis"""
        
        issues = []
        
        # 1. Stance Analysis
        stance_issues = self._analyze_stance(keypoints)
        issues.extend(stance_issues)
        
        # 2. Batting Arm Analysis
        arm_issues = self._analyze_batting_arm(angles, keypoints)
        issues.extend(arm_issues)
        
        # 3. Body Positioning Analysis
        body_issues = self._analyze_body_position(keypoints, angles)
        issues.extend(body_issues)
        
        # 4. Balance Analysis
        balance_issues = self._analyze_balance(keypoints)
        issues.extend(balance_issues)
        
        # 5. Overall Form Assessment
        form_issues = self._assess_overall_form(keypoints, angles)
        issues.extend(form_issues)
        
        return issues
    
    def _analyze_stance(self, keypoints: Dict) -> List[CricketIssue]:
        """Analyze batting stance"""
        issues = []
        
        if not all(key in keypoints for key in ['LEFT_ANKLE', 'RIGHT_ANKLE']):
            return issues
        
        left_foot = keypoints['LEFT_ANKLE']
        right_foot = keypoints['RIGHT_ANKLE']
        
        # Calculate stance width (normalized)
        stance_width = abs(left_foot[0] - right_foot[0])
        
        if stance_width < self.optimal_ranges['stance_width'][0]:
            issues.append(CricketIssue(
                type="Narrow Stance",
                description="Your stance is too narrow, limiting balance and power generation",
                severity="High",
                recommendation="Widen your feet to shoulder-width apart for better stability and power transfer",
                confidence=0.85
            ))
        elif stance_width > self.optimal_ranges['stance_width'][1]:
            issues.append(CricketIssue(
                type="Wide Stance", 
                description="Your stance is too wide, reducing mobility and shot flexibility",
                severity="Medium",
                recommendation="Bring your feet closer together for better mobility while maintaining balance",
                confidence=0.8
            ))
        
        # Check foot alignment
        foot_height_diff = abs(left_foot[1] - right_foot[1])
        if foot_height_diff > 0.05:  # normalized threshold
            issues.append(CricketIssue(
                type="Uneven Foot Placement",
                description="Your feet are not level, affecting balance and shot execution",
                severity="Medium", 
                recommendation="Ensure both feet are planted firmly and level on the ground",
                confidence=0.75
            ))
        
        return issues
    
    def _analyze_batting_arm(self, angles: Dict, keypoints: Dict) -> List[CricketIssue]:
        """Analyze batting arm technique"""
        issues = []
        
        if 'batting_arm' not in angles:
            return issues
        
        arm_angle = angles['batting_arm']
        
        if arm_angle < self.optimal_ranges['batting_arm_angle'][0]:
            issues.append(CricketIssue(
                type="Collapsed Batting Arm",
                description="Your batting arm is too bent, reducing reach and power",
                severity="High",
                recommendation="Extend your batting arm more to create a longer arc and generate more power",
                confidence=0.9
            ))
        elif arm_angle > self.optimal_ranges['batting_arm_angle'][1]:
            issues.append(CricketIssue(
                type="Over-extended Arm",
                description="Your batting arm is too straight, potentially causing timing issues",
                severity="Medium", 
                recommendation="Slightly bend your elbow to maintain control and timing flexibility",
                confidence=0.8
            ))
        
        # Check wrist position
        if all(key in keypoints for key in ['RIGHT_ELBOW', 'RIGHT_WRIST']):
            elbow = keypoints['RIGHT_ELBOW']
            wrist = keypoints['RIGHT_WRIST']
            
            # Simple wrist drop check
            if wrist[1] > elbow[1] + 0.1:  # wrist significantly below elbow
                issues.append(CricketIssue(
                    type="Dropped Wrists",
                    description="Your wrists are dropped, affecting bat control and power transfer", 
                    severity="Medium",
                    recommendation="Keep your wrists firm and level with your elbows for better bat control",
                    confidence=0.75
                ))
        
        return issues
    
    def _analyze_body_position(self, keypoints: Dict, angles: Dict) -> List[CricketIssue]:
        """Analyze body positioning and alignment"""
        issues = []
        
        # Shoulder alignment check
        if all(key in keypoints for key in ['LEFT_SHOULDER', 'RIGHT_SHOULDER']):
            left_shoulder = keypoints['LEFT_SHOULDER'] 
            right_shoulder = keypoints['RIGHT_SHOULDER']
            
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_angle = np.degrees(np.arctan(shoulder_diff / (abs(left_shoulder[0] - right_shoulder[0]) + 1e-6)))
            
            if shoulder_angle > self.optimal_ranges['shoulder_alignment'][1]:
                issues.append(CricketIssue(
                    type="Shoulder Misalignment",
                    description="Your shoulders are not level, affecting shot accuracy and balance",
                    severity="High",
                    recommendation="Keep your shoulders level and parallel to maintain proper batting form",
                    confidence=0.85
                ))
        
        # Hip alignment check  
        if all(key in keypoints for key in ['LEFT_HIP', 'RIGHT_HIP']):
            left_hip = keypoints['LEFT_HIP']
            right_hip = keypoints['RIGHT_HIP']
            
            hip_diff = abs(left_hip[1] - right_hip[1])
            if hip_diff > 0.08:  # normalized threshold
                issues.append(CricketIssue(
                    type="Hip Misalignment", 
                    description="Your hips are not level, affecting weight transfer and power generation",
                    severity="Medium",
                    recommendation="Align your hips to ensure proper weight transfer during shots",
                    confidence=0.8
                ))
        
        # Body lean analysis
        if 'body_lean' in angles:
            if angles['body_lean'] > self.optimal_ranges['body_lean'][1]:
                issues.append(CricketIssue(
                    type="Excessive Body Lean",
                    description="You're leaning too much, affecting balance and shot execution",
                    severity="Medium",
                    recommendation="Maintain a more upright posture with slight forward lean",
                    confidence=0.8
                ))
        
        return issues
    
    def _analyze_balance(self, keypoints: Dict) -> List[CricketIssue]:
        """Analyze overall balance and weight distribution"""
        issues = []
        
        # Center of mass analysis (simplified)
        if all(key in keypoints for key in ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']):
            
            # Calculate approximate center of mass
            upper_center_x = (keypoints['LEFT_SHOULDER'][0] + keypoints['RIGHT_SHOULDER'][0]) / 2
            lower_center_x = (keypoints['LEFT_HIP'][0] + keypoints['RIGHT_HIP'][0]) / 2
            
            lateral_shift = abs(upper_center_x - lower_center_x)
            
            if lateral_shift > 0.1:  # normalized threshold
                issues.append(CricketIssue(
                    type="Poor Weight Distribution",
                    description="Your weight is not centered, affecting stability and shot power",
                    severity="Medium",
                    recommendation="Focus on keeping your weight centered between both feet",
                    confidence=0.7
                ))
        
        return issues
    
    def _assess_overall_form(self, keypoints: Dict, angles: Dict) -> List[CricketIssue]:
        """Overall batting form assessment"""
        issues = []
        
        # Count high-severity issues
        high_severity_count = sum(1 for issue in issues if issue.severity == "High")
        
        if high_severity_count == 0 and len(issues) <= 1:
            issues.append(CricketIssue(
                type="Excellent Form",
                description="Your batting technique shows excellent fundamentals!",
                severity="Good",
                recommendation="Continue practicing to maintain this excellent form",
                confidence=0.9
            ))
        elif high_severity_count >= 3:
            issues.append(CricketIssue(
                type="Multiple Technical Issues",
                description="Several aspects of your technique need attention",
                severity="High", 
                recommendation="Work with a coach to address fundamental batting mechanics systematically",
                confidence=0.85
            ))
        
        return issues
    
    def get_performance_score(self, issues: List[CricketIssue]) -> Tuple[int, str]:
        """Calculate overall performance score out of 100"""
        
        if not issues:
            return 95, "Excellent"  # Changed from 90
        
        # LESS HARSH scoring system
        score = 100
        for issue in issues:
            if issue.severity == "High":
                score -= 10  # Was 15 - reduced penalty
            elif issue.severity == "Medium": 
                score -= 5   # Was 8 - reduced penalty
            elif issue.severity == "Low":
                score -= 2   # Was 3 - reduced penalty
        
        score = max(score, 0)
        
        # ADJUSTED grade thresholds
        if score >= 80:         # Was 85
            grade = "Excellent"
        elif score >= 65:       # Was 70
            grade = "Good" 
        elif score >= 50:       # Was 55
            grade = "Average"
        else:
            grade = "Needs Improvement"
        
        return score, grade

