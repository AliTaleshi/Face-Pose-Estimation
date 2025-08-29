import cv2
import numpy as np
from math import cos, sin
import pickle
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from src.preprocessing import preprocess
from src.visualization import draw_axis_webcam

with open('./output/best_model.pkl', 'rb') as f:
    multi_output_regressor = pickle.load(f)

faceModule = mp_face_mesh

def get_face_center(face_landmarks, width, height):
    nose_x = int(face_landmarks.landmark[1].x * width)
    nose_y = int(face_landmarks.landmark[1].y * height)
    return nose_x, nose_y

def get_face_direction(pitch, yaw, roll):
    direction = []
    
    if yaw > 15:
        direction.append("Left")
    elif yaw < -15:
        direction.append("Right")
    else:
        direction.append("Center")
    
    if pitch > 15:
        direction.append("Up")
    elif pitch < -15:
        direction.append("Down")
    else:
        direction.append("Forward")
    
    if abs(roll) > 15:
        if roll > 0:
            direction.append("Tilt-Right")
        else:
            direction.append("Tilt-Left")
    
    return " ".join(direction)

def webcam_face_pose_estimation():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Trying alternative camera indices...")
        for i in range(1, 3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Successfully opened camera index {i}")
                break
        if not cap.isOpened():
            print("Failed to open any camera")
            return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with faceModule.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("Starting webcam face pose estimation...")
        print("Controls:")
        print("- Press 'q' or 'Q' to quit")
        print("- Press 's' or 'S' to save current frame")
        print("- Press 'h' or 'H' to toggle help display")
        print("- Press 'ESC' to quit")
        print("- Make sure the OpenCV window is focused (click on it)")
        print("- If keys don't work, click on the video window first")
        
        cv2.namedWindow('Face Pose Estimation - Webcam', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        show_help = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam")
                break
            
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(rgb_frame)
            
            frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    height, width, _ = frame_bgr.shape
                    
                    try:
                        processed_landmarks = preprocess(face_landmarks, width, height)
                        
                        angles = multi_output_regressor.predict(processed_landmarks.reshape(1, -1))
                        pitch, yaw, roll = angles[0, 0], angles[0, 1], angles[0, 2]
                        
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)
                        roll_deg = np.degrees(roll)
                        
                        face_center_x, face_center_y = get_face_center(face_landmarks, width, height)
                        
                        frame_bgr = draw_axis_webcam(frame_bgr, pitch, yaw, roll, 
                                                   face_center_x, face_center_y, size=80)
                        
                        direction = get_face_direction(pitch_deg, yaw_deg, roll_deg)
                        
                        info_text = [
                            f"Pitch: {pitch_deg:.1f} deg",
                            f"Yaw: {yaw_deg:.1f} deg", 
                            f"Roll: {roll_deg:.1f} deg",
                            f"Direction: {direction}"
                        ]
                        
                        for i, text in enumerate(info_text):
                            cv2.putText(frame_bgr, text, (10, 30 + i * 25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        cv2.putText(frame_bgr, "Prediction Error", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame_bgr, "No face detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame_bgr, "Position your face in the camera view", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if show_help:
                help_text = ["Controls: 'q'/'Q'=quit, 's'/'S'=save, 'h'/'H'=help, 'ESC'=quit",
                           "Red=X-axis, Green=Y-axis, Blue=Z-axis",
                           "Click on this window to enable key controls"]
                for i, text in enumerate(help_text):
                    cv2.putText(frame_bgr, text, (10, height - 60 + i * 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            cv2.imshow('Face Pose Estimation - Webcam', frame_bgr)
            
            cv2.setWindowProperty('Face Pose Estimation - Webcam', cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1) & 0xFF
            
            if key != 255:
                print(f"Key pressed: {key} ('{chr(key)}' if valid)")
            
            if key == ord('q') or key == ord('Q'):
                print("Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                filename = f"pose_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame_bgr)
                print(f"Frame saved as {filename}")
                frame_count += 1
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help
                print(f"Help display: {'ON' if show_help else 'OFF'}")
            elif key == 27:
                print("ESC pressed - Quitting...")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam face pose estimation stopped.")

def start_webcam_pose_estimation():
    try:
        webcam_face_pose_estimation()
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your webcam connection and try again.")
