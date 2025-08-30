import numpy as np
import scipy.io as sio
import glob
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh
face_mesh_module = mp_face_mesh.FaceMesh(static_image_mode=True)

def map_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def preprocess(face_landmarks, width=450, height=450):
    landmarks_to_use = face_landmarks.landmark[:468]
    
    x_val = [lm.x * width for lm in landmarks_to_use]
    y_val = [lm.y * height for lm in landmarks_to_use]
    
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    
    x_val = x_val - x_val[1]
    y_val = y_val - y_val[1]
    
    if x_val.max() != 0:
        x_val = x_val / x_val.max()
    if y_val.max() != 0:
        y_val = y_val / y_val.max()
    
    return np.concatenate([x_val, y_val])


def load_aflw2000(dataset_dir: str = "./AFLW2000"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_module = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    files = glob.glob(f'{dataset_dir}/*.jpg')
    print(f"📂 Found {len(files)} images in dataset")

    images, landmarks, pitch, yaw, roll = [], [], [], [], []

    with mp_face_mesh.FaceMesh(static_image_mode=True) as faces:
        for file in files:
            image = cv2.imread(file)
            if image is None:
                continue

            mat_data = sio.loadmat(file.replace('jpg', 'mat'))
            if mat_data is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = faces.process(image_rgb)

            if results.multi_face_landmarks is None:
                continue

            h, w = image_rgb.shape[:2]
            x_val = [lm.x * w for lm in results.multi_face_landmarks[0].landmark]
            y_val = [lm.y * h for lm in results.multi_face_landmarks[0].landmark]

            x_val = np.array(x_val) - np.mean(x_val[1])
            y_val = np.array(y_val) - np.mean(y_val[1])

            x_val /= x_val.max()
            y_val /= y_val.max()

            pose = mat_data["Pose_Para"][0][:3]
            landmarks.append(np.concatenate([x_val, y_val]))
            pitch.append(pose[0])
            yaw.append(pose[1])
            roll.append(pose[2])
            images.append(Path(file).stem)

    df = pd.DataFrame({
        'Image_Id': images,
        'Landmarks': landmarks,
        'Pitch': pitch,
        'Yaw': yaw,
        'Roll': roll
    })

    df[['Pitch', 'Yaw', 'Roll']] = df[['Pitch', 'Yaw', 'Roll']].apply(map_to_pi)
    print("✅ Dataset prepared:", df.shape)
    X = np.array(df['Landmarks'].to_list())
    y = np.array(df[['Pitch', 'Yaw', 'Roll']])
    return X, y
