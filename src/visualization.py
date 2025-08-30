import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy import stats
from sklearn.metrics import r2_score
from math import cos, sin
from src.preprocessing import face_mesh_module, preprocess


def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, scale_factor=0.2, thickness_factor=0.015):
    yaw = -yaw
    h, w = img.shape[:2]

    size = int(min(w, h) * scale_factor)
    thickness = max(1, int(min(w, h) * thickness_factor))

    if tdx is None or tdy is None:
        tdx, tdy = w / 2, h / 2

    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness)  # X
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness)  # Y
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness)  # Z
    return img

def draw_axis_webcam(img, pitch, yaw, roll, tdx=None, tdy=None, size=100):
    yaw = -yaw
    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)  
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)
    
    cv2.circle(img, (int(tdx), int(tdy)), 5, (255, 255, 255), -1)
    
    return img

def plot_regression_results(y_true, y_pred, metrics, targets=["Pitch", "Yaw", "Roll"]):
    residuals = y_true.flatten() - y_pred.flatten()
    abs_errors = np.abs(residuals)

    fig = plt.figure(figsize=(20, 20))

    plt.subplot(3, 3, 1)
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.6, s=30)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted\nR² = {metrics["R2"]:.3f}')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 2)
    plt.scatter(y_pred.flatten(), residuals, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 3)
    plt.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 5)
    metric_names = ['R²', 'RMSE', 'MAE']
    metric_values = [metrics['R2'], metrics['RMSE'], metrics['MAE']]
    bars = plt.bar(metric_names, metric_values, color=['green', 'orange', 'blue'], alpha=0.7)
    plt.title('Key Regression Metrics')
    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                 ha='center', va='bottom')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 6)
    plt.scatter(y_pred.flatten(), abs_errors, alpha=0.6, s=30)
    plt.title('Absolute Error vs Predictions')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 7)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
        plt.bar(range(len(scores)), scores, alpha=0.7, color='purple')
        plt.xticks(range(len(scores)), targets)
        for i, score in enumerate(scores):
            plt.text(i, score + 0.01, f'{score:.3f}', ha='center')
        plt.title('R² by Output Dimension')
    else:
        plt.plot(y_true.flatten(), label="Actual")
        plt.plot(y_pred.flatten(), label="Predicted")
        plt.legend()
        plt.title("Predictions vs Actual")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 8)
    pred_range = np.linspace(y_true.min(), y_true.max(), 100)
    plt.plot(pred_range, pred_range, 'r--', label='Perfect Prediction')
    plt.scatter(y_true.flatten()[:100], y_pred.flatten()[:100], alpha=0.6, label='Predictions')
    plt.title('Prediction Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Comprehensive Regression Model Evaluation', fontsize=16, y=0.98)
    plt.show()

def inference_on_image(image_path=None):
    if image_path is None:
        root = tk.Tk()
        root.withdraw()
        
        image_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if not image_path:
            print("No file selected. Exiting.")
            return
    
    with open("output/best_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    image = cv2.imread(image_path)

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_module.process(image_rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            width, height = image_rgb.shape[:2]

            angles = best_model.predict(preprocess(face, width, height).reshape(1,-1))
            pitch, yaw, roll = angles[0,0], angles[0,1], angles[0,2]
            center = face.landmark[1]
            print('Predicted values: ', pitch, yaw, roll)
            
            image_with_axes = draw_axis(image_rgb.copy(), pitch, yaw, roll, tdx=center.x * height, tdy=center.y * width)

            plt.figure(figsize=(10, 8))
            plt.imshow(image_with_axes)
            plt.axis('off')
            plt.title(f'Pose: Pitch={pitch:.2f}, Yaw={yaw:.2f}, Roll={roll:.2f}')
            plt.tight_layout()
            plt.show()

    else:
        print(f"Error: Unable to load the image from {image_path}.")
