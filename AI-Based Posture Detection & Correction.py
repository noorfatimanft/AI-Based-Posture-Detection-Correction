import cv2
import mediapipe as mp
import numpy as np
import time
import platform
import os
import statistics
from dataclasses import dataclass
from typing import List, Tuple, Optional


def beep(freq=1000, duration=200):
    
    system_name = platform.system()
    
    if system_name == "Windows":
        try:
            import winsound
            winsound.Beep(freq, duration)
        except ImportError:
            print('\a') 
    elif system_name == "Darwin": 
        print('\a')
    else: 
        print('\a') 


class Config:
    
    COLOR_GOOD = (0, 255, 0)      
    COLOR_WARN = (0, 255, 255)    
    COLOR_BAD = (0, 0, 255)       
    COLOR_TEXT = (255, 255, 255)  
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    CALIBRATION_TIME = 5.0  


def calculate_angle_3d(a, b, c):
   
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_inclination_sagittal(p1, p2):
    
    p1 = np.array(p1)
    p2 = np.array(p2)

    v = np.array([
        0,
        p2[1] - p1[1],  
        p2[2] - p1[2]   
    ])

    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return 0.0

    unit_v = v / norm_v

    vertical = np.array([0, 1, 0])

    dot = np.dot(unit_v, vertical)
    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    return angle


class PostureAssistant:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        
        self.calibrated = False
        self.calibration_start_time = None
        self.calibration_samples = []
        
        
        self.baseline_neck_angle = 180
        self.baseline_torso_angle = 180
        self.baseline_shoulder_ear_z_dist = 0
        
        
        self.bad_posture_start_time = None
        
        self.status_message = "Press 'c' to Calibrate"
        self.color = Config.COLOR_WARN
        
        self.metrics = {}

    def get_landmarks_3d(self, results):
        if not results.pose_world_landmarks:
            return None
        
        lm = results.pose_world_landmarks.landmark
        P = self.mp_pose.PoseLandmark
        
        def get_pt(idx):
            return [lm[idx].x, lm[idx].y, lm[idx].z]

        nose = get_pt(P.NOSE)
        ear_l = get_pt(P.LEFT_EAR)
        ear_r = get_pt(P.RIGHT_EAR)
        shoulder_l = get_pt(P.LEFT_SHOULDER)
        shoulder_r = get_pt(P.RIGHT_SHOULDER)
        hip_l = get_pt(P.LEFT_HIP)
        hip_r = get_pt(P.RIGHT_HIP)
        
        
        ear_z = (ear_l[2] + ear_r[2]) / 2
        ear_mid = [(ear_l[0]+ear_r[0])/2, (ear_l[1]+ear_r[1])/2, ear_z]
        
        shoulder_z = (shoulder_l[2] + shoulder_r[2]) / 2
        shoulder_mid = [(shoulder_l[0]+shoulder_r[0])/2, (shoulder_l[1]+shoulder_r[1])/2, shoulder_z]
        
        hip_z = (hip_l[2] + hip_r[2]) / 2
        hip_mid = [(hip_l[0]+hip_r[0])/2, (hip_l[1]+hip_r[1])/2, hip_z]
        
        return {
            'nose': nose,
            'ear': ear_mid,
            'shoulder': shoulder_mid,
            'hip': hip_mid,
            'ear_l': ear_l, 'ear_r': ear_r,
            'shoulder_l': shoulder_l, 'shoulder_r': shoulder_r
        }

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        display_frame = frame.copy()
        
        if not results.pose_landmarks:
            cv2.putText(display_frame, "No person detected", (50, h//2), Config.FONT, 1, Config.COLOR_BAD, 2)
            return display_frame

        pts = self.get_landmarks_3d(results)
        if not pts:
            return display_frame

        
        neck_angle = calculate_angle_3d(pts['ear'], pts['shoulder'], pts['hip'])
        torso_inclination = calculate_inclination_sagittal(pts['hip'], pts['shoulder'])
        
        
        shoulder_hip_z = pts['shoulder'][2] - pts['hip'][2]
        
        self.metrics = {
            'neck_angle': neck_angle,
            'torso_inclination': torso_inclination,
            'shoulder_hip_z': shoulder_hip_z
        }

        
        if not self.calibrated:
            if self.calibration_start_time is None:
                self.color = Config.COLOR_WARN
                self.status_message = "Press 'c' to Start Calibration"
                self.instruction_message = ""
            else:
                elapsed = time.time() - self.calibration_start_time
                if elapsed < Config.CALIBRATION_TIME:
                    self.status_message = f"Calibrating... {int(Config.CALIBRATION_TIME - elapsed)}"
                    self.instruction_message = "Sit Up Straight!"
                    self.color = Config.COLOR_WARN
                    self.calibration_samples.append(self.metrics)
                else:
                    self.complete_calibration()
        else:
            self.evaluate_posture(self.metrics)

       
        self.draw_hud(display_frame, self.metrics, pts, results.pose_landmarks)
        
        return display_frame

    def start_calibration(self):
        self.calibration_start_time = time.time()
        self.calibration_samples = []
        self.calibrated = False
        self.instruction_message = ""

    def complete_calibration(self):
        n = len(self.calibration_samples)
        if n == 0:
           return

        self.baseline_neck_angle = statistics.median(
            m['neck_angle'] for m in self.calibration_samples
    )

        self.baseline_torso_angle = statistics.median(
            m['torso_inclination'] for m in self.calibration_samples
    )

        self.baseline_shoulder_hip_z = statistics.median(
            m['shoulder_hip_z'] for m in self.calibration_samples
    )

        self.calibrated = True
        self.status_message = "Monitoring..."
        self.instruction_message = "Keep this posture"

        print(
            f"Baseline -> Neck: {self.baseline_neck_angle:.1f}, "
            f"Torso: {self.baseline_torso_angle:.1f}, "
            f"Sh-Hip-Z: {self.baseline_shoulder_hip_z:.3f}"
    )

    def evaluate_posture(self, metrics):
        issues = []
        instructions = []
        is_severe = False
        
        
        neck_diff = self.baseline_neck_angle - metrics['neck_angle']
        if neck_diff > 12: 
            issues.append("Neck")
            instructions.append("Pull Head Back")
            is_severe = True
        elif neck_diff > 7: 
            issues.append("Neck")
            instructions.append("Fix Neck")
            
         
        torso_diff = abs(metrics['torso_inclination'] - self.baseline_torso_angle)
        if torso_diff > 6:
             issues.append("Lean")
             instructions.append("Sit Upright")
             is_severe = True
        elif torso_diff > 2.5:
             issues.append("Lean")
             instructions.append("Adjust Torso")
             
        
        z_diff = metrics['shoulder_hip_z'] - self.baseline_shoulder_hip_z
        
    
        if z_diff < -0.02: 
             issues.append("Slouch")
             instructions.append("Pull Shoulders Back")
             is_severe = True
        elif z_diff < -0.008: 
             issues.append("Slouch")
             instructions.append("Open Chest")

        if not issues:
            self.status_message = "POSTURE: GOOD"
            self.instruction_message = "Great!"
            self.color = Config.COLOR_GOOD
            self.bad_posture_start_time = None
        else:
            self.status_message = "BAD POSTURE: " + ", ".join(set(issues))
            self.instruction_message = " & ".join(set(instructions))
            
            if is_severe:
                self.color = Config.COLOR_BAD
            else:
                self.color = Config.COLOR_WARN
            
            
            now = time.time()
            if self.bad_posture_start_time is None:
                self.bad_posture_start_time = now
            elif now - self.bad_posture_start_time > 2.0:
                freq = 1000 if is_severe else 600
                beep(freq, 250) 
                self.bad_posture_start_time = now

    def draw_hud(self, frame, metrics, pts_3d, pose_landmarks_2d):
        h, w, _ = frame.shape
        
        
        valid_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28)
        ]
        
        color = (255, 255, 255)
        if self.color == Config.COLOR_BAD: color = (200, 200, 255) 
        
        for start_idx, end_idx in valid_connections:
            if start_idx >= len(pose_landmarks_2d.landmark) or end_idx >= len(pose_landmarks_2d.landmark): continue
            p1 = pose_landmarks_2d.landmark[start_idx]
            p2 = pose_landmarks_2d.landmark[end_idx]
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), color, 2)
        
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
       
        cv2.putText(frame, self.instruction_message, (20, 50), Config.FONT, 0.8, self.color, 2)
        cv2.putText(frame, self.status_message, (20, 90), Config.FONT, 0.6, (255, 255, 255), 1)
        
        if self.calibrated:
            row_metrics = f"Neck:{int(metrics['neck_angle'])}  Z-Delta:{int((metrics['shoulder_hip_z'] - self.baseline_shoulder_hip_z)*100)}cm"
            cv2.putText(frame, row_metrics, (20, 120), Config.FONT, 0.5, (200, 200, 200), 1)
        else:
             cv2.putText(frame, "Waiting for Calibration...", (20, 120), Config.FONT, 0.5, (200, 200, 200), 1)


def main():
    
    try:
        mp.solutions
    except AttributeError:
        print("\nCRITICAL ERROR: 'mediapipe' has no attribute 'solutions'.")
        print("Possible causes:")
        print("1. You named a file 'mediapipe.py' in this folder. Rename it!")
        print("2. MediaPipe is not installed correctly.")
        print("   Run: pip uninstall mediapipe && pip install mediapipe")
        print("3. On macOS M1/M2, you might need: pip install mediapipe-silicon")
        return


    
    print("--- AI Posture Assistant ---")
    print("Initializing Camera...")
    
    cap = cv2.VideoCapture(0)
    
    
    if not cap.isOpened():
        print("Camera index 0 failed. Trying index 1...")
        cap.release()
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    
    assistant = PostureAssistant()
    
    cv2.namedWindow('AI Posture Assistant', cv2.WINDOW_NORMAL)
    
    print("1. Sit comfortably and straight.")
    print("2. Press 'c' to Calibrate (5 seconds).")
    print("3. The system will alert you if you slouch.")
    print("Press 'q' to Quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
            
        frame = cv2.flip(frame, 1) 
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            assistant.start_calibration()
            
    
        output_frame = assistant.process_frame(frame)
        
        cv2.imshow('AI Posture Assistant', output_frame)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()