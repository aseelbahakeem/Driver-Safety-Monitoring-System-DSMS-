"""
PERCLOS-Based Drowsiness Detection System
Real-time Eye Aspect Ratio (EAR) tracking for driver safety
"""

import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque
import mediapipe as mp
import time

class PERCLOSDetector:
    """
    Temporal drowsiness detection using PERCLOS (Percentage of Eye Closure)
    """
    
    def __init__(self):
        print("Initializing PERCLOS Detector...")
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices (MediaPipe Face Mesh)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Temporal tracking parameters
        self.EAR_HISTORY = deque(maxlen=90)  # 3 seconds at 30 FPS
        self.EAR_THRESHOLD = 0.25
        self.DROWSY_THRESHOLD = 30  # PERCLOS > 30% = drowsy
        
        # Statistics
        self.frame_count = 0
        self.alert_frames = 0
        self.drowsy_frames = 0
        self.start_time = time.time()
        
        print("PERCLOS Detector initialized!")
        
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Vertical distances
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR calculation
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, face_landmarks, eye_indices, frame_shape):
        """Extract eye landmark coordinates from MediaPipe results"""
        h, w = frame_shape[:2]
        landmarks = []
        
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks
    
    def process_frame(self, frame):
        """
        Process single frame and return drowsiness status
        """
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        # No face detected
        if not results.multi_face_landmarks:
            status = "No Face Detected"
            color = (0, 0, 255)  # Red
            return frame, status, color, 0, 0
        
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract eye landmarks
        left_eye = self.extract_eye_landmarks(
            face_landmarks, self.LEFT_EYE, frame.shape
        )
        right_eye = self.extract_eye_landmarks(
            face_landmarks, self.RIGHT_EYE, frame.shape
        )
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Update temporal history
        self.EAR_HISTORY.append(avg_ear)
        
        # Calculate PERCLOS (only after enough frames)
        if len(self.EAR_HISTORY) >= 30:  # At least 1 second of data
            closed_frames = sum(1 for ear in self.EAR_HISTORY 
                              if ear < self.EAR_THRESHOLD)
            perclos = (closed_frames / len(self.EAR_HISTORY)) * 100
        else:
            perclos = 0
        
        # Determine drowsiness status
        if perclos > self.DROWSY_THRESHOLD:
            status = "drowsy"
            color = (0, 165, 255)  # Orange
            self.drowsy_frames += 1
        else:
            status = "alert"
            color = (0, 255, 0)  # Green
            self.alert_frames += 1
        
        # Draw eye landmarks
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # Display information on frame
        cv2.putText(frame, status, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add threshold indicator
        threshold_text = f"Threshold: {self.DROWSY_THRESHOLD}%"
        cv2.putText(frame, threshold_text, (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Add visual border if drowsy
        if perclos > self.DROWSY_THRESHOLD:
            cv2.rectangle(frame, (0, 0), 
                         (frame.shape[1], frame.shape[0]), 
                         color, 15)
        
        return frame, status, color, avg_ear, perclos
    
    def print_statistics(self):
        """Print session statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
     
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Session Duration: {elapsed_time:.1f} seconds")
        print(f"Average FPS: {fps:.1f}")
        print(f"\nAlert Frames: {self.alert_frames} "
              f"({self.alert_frames/self.frame_count*100:.1f}%)")
        print(f"Drowsy Frames: {self.drowsy_frames} "
              f"({self.drowsy_frames/self.frame_count*100:.1f}%)")


def main():
    print("\nInstructions:")
    print("1. Position your face in front of the camera")
    print("2. Keep eyes OPEN normally for 10 seconds")
    print("3. CLOSE your eyes for 5 seconds (simulate drowsiness)")
    print("4. Open eyes again to see status change back")
    print("\nPress 'q' to quit")
    
    # Initialize detector
    detector = PERCLOSDetector()
    
    # Open webcam
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return
    
    print("Camera opened successfully!")
    print("Starting detection...\n")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            # Process frame
            processed_frame, status, color, ear, perclos = detector.process_frame(frame)
            
            # Display frame
            cv2.imshow('PERCLOS Drowsiness Detection', processed_frame)
            
            # Print status every 30 frames
            if detector.frame_count % 30 == 0:
                print(f"Frame {detector.frame_count:4d} | "
                      f"Status: {status:20s} | "
                      f"EAR: {ear:.3f} | "
                      f"PERCLOS: {perclos:5.1f}%")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping detection...")
                break
                
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        detector.print_statistics()
        
        print("\nDetection session completed!")


if __name__ == "__main__":
    main()