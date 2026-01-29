AI Posture Assistant

Real-Time Posture Monitoring Using Computer Vision & MediaPipe

Overview :

The AI Posture Assistant is a real-time computer vision application that helps users maintain good sitting posture while working on a computer.
It uses your webcam, MediaPipe Pose, and OpenCV to detect body landmarks and continuously analyze neck, torso, and shoulder alignment.
When poor posture is detected for a certain duration, the system visually warns the user and also gives audio alerts to encourage correction.

This project is especially useful for:

1.Students
2.Office workers
3.Programmers

Key Features :

-Live webcam posture tracking
-AI-based pose estimation using MediaPipe
-3D angle calculations for neck and torso
-Personalized posture calibration

Real-time posture feedback :

Good posture (Green)
Warning (Yellow)
Severe posture (Red)

Audio alert (beep) when posture remains bad

Clean on-screen HUD (instructions + metrics)

Technologies Used :

Python
OpenCV – video processing & visualization
MediaPipe Pose – human pose detection
NumPy – vector & angle calculations
Statistics module – robust baseline calibration
Platform-specific audio alerts

How It Works :
The webcam captures live video frames.
MediaPipe detects body landmarks (ears, shoulders, hips).

The system calculates:

Neck angle
Torso inclination
Shoulder-hip depth alignment
You calibrate once while sitting straight.      
Your posture is compared with the calibrated baseline.

If posture worsens:
On-screen instructions appear
Beep sound alerts you after a short delay

Controls (Key	Action) :
c	Start posture calibration (5 seconds)
q	Quit the application
Esc	Quit the application

Installation & Setup :
1- Clone the Repository
git clone https://github.com/your-username/ai-posture-assistant.git
cd ai-posture-assistant

2- Install Required Libraries
pip install opencv-python mediapipe numpy
For Apple Silicon (M1 / M2):
pip install mediapipe-silicon

Make sure:
Your webcam is connected
No other app is using the camera

Calibration Instructions (Important) :

Sit straight and comfortably
Face the camera naturally
Press c
Stay still for 5 seconds
The system starts monitoring automatically
Calibration helps the system adapt to your body posture, not a generic one.

On-Screen Information

Instruction Message – what you should correct
Posture Status – GOOD / BAD posture
Neck Angle – posture angle value
Z-Delta – shoulder slouch depth difference

Audio Alerts

Soft beep → mild posture issue
Loud beep → severe posture issue
Alerts trigger only if bad posture persists (not instantly)

This avoids annoying false alarms 

Learning Outcomes

This project demonstrates:

Real-time computer vision
Pose estimation
3D geometry & vector math
Human-centered AI design
Practical use of AI for health & ergonomics
