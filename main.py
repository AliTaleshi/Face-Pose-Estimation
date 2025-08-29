# =========================================
# 1. Project Setup and Imports
# =========================================
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import load_aflw2000, preprocess
from src.metrics import calculate_regression_metrics
from src.visualization import plot_regression_results, inference_on_image
from src.models import train_and_select_best
from src.webcam import start_webcam_pose_estimation


# =========================================
# 2. Load and Preprocess AFLW2000 Dataset
# =========================================
dataset_dir = "./dataset/AFLW2000"
X, y = load_aflw2000(dataset_dir)


# =========================================
# 3. Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================
# 4. Train Models and Select Best
# =========================================
best_model, best_name = train_and_select_best(X_train, y_train, X_test, y_test)

# Save the best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# =========================================
# 5. Evaluate Best Model
# =========================================
y_pred = best_model.predict(X_test)

metrics = calculate_regression_metrics(y_test, y_pred)
print("📊 Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

plot_regression_results(y_test, y_pred, metrics, targets=["Pitch", "Yaw", "Roll"])

# =========================================
# 6. Run Inference on New Image
# =========================================
inference_on_image()

# =========================================
# 7. Run Inference on Webcam
# =========================================
start_webcam_pose_estimation()
