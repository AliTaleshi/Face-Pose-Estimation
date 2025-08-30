# Face-Pose-Estimation

Head pose estimation using **MediaPipe FaceMesh** and **Support Vector Regression (SVR)** — supports dataset-based evaluation and real-time webcam inference with **pitch**, **yaw**, and **roll** predictions.

---

##  Project Structure

```
Face-Pose-Estimation/
│
├── main.py # (Optional) Entrypoint for manual running
│
├── src/ # Source code modules
│ ├── preprocessing.py # Data loading, landmark extraction, normalization
│ ├── metrics.py # Regression metrics calculation
│ ├── models.py # Model training, evaluation, saving/loading logic
│ ├── visualization.py # Draw axis, plots, image inference
│ └── webcam.py # Real-time webcam head pose estimation
│
├── notebooks/
│ └── face_pose_pipeline.ipynb # Main Jupyter Notebook (experiments & workflow)
│
├── dataset/ # Dataset directory (ignored in repo)
│ └── AFLW2000/ # Contains .jpg and .mat files (not included in repo)
│
├── output/ # Model results & saved models (ignored in repo)
│ └── *.pkl / *.txt
│
├── image.jpg # Test image for model performance evaluation
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore                     # This file
```

> **Note**: The `output/` directory (containing model files, result logs, plots) and the `dataset/AFLW2000/` directory are excluded from this repository for file size reasons.

---

##  Quick Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AliTaleshi/Face-Pose-Estimation.git
   cd Face-Pose-Estimation
   ```

2. **Install dependencies**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Getting Started (Kaggle/Colab Setup)

If you are running this project on **Google Colab** or using **Kaggle datasets**, follow these steps to install dependencies and prepare the datasets:

```bash
# Install dependencies
!pip install mediapipe

# Setup Kaggle API credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download AFLW2000-3D dataset
!kaggle datasets download mohamedadlyi/aflw2000-3d
!unzip aflw2000-3d.zip

# Download CelebA dataset (optional for face images)
!kaggle datasets download jessicali9530/celeba-dataset
!unzip celeba-dataset.zip
```

---

##  Usage Guide

### 1. Training & Evaluation (Jupyter Notebook)
Launch the notebook to run the full head pose estimation pipeline:
```bash
jupyter notebook face-pose-pipeline.ipynb
```
This notebook covers:
- Loading and preprocessing dataset
- Training SVR (and optionally other regressors)
- Evaluating using regression metrics
- Visualizing results and saving model artifacts
- Inference on image and webcam

---

### 2. Image Inference
You can use `main.py` to manually run the project (Optional):
```bash
python main.py
```

Example usage inside `main.py` (pseudocode):
```python
from src.preprocessing import inference_on_image

# =========================================
# 6. Run Inference on New Image
# =========================================
inference_on_image()
```

---

### 3. Real-Time Webcam Inference
For live head pose estimation with axis overlay:
```bash
python main.py
```

Example usage inside `main.py` (pseudocode):
```python
from src.webcam import start_webcam_pose_estimation()

# =========================================
# 7. Run Inference on Webcam
# =========================================
start_webcam_pose_estimation()
```

**Controls**:
- `Q` → Quit  
- `S` → Save frame  
- `H` → Toggle help text overlay  
- `Esc` → Quit  

Colored lines on the face represent the 3D axes:
- **Red**: X-axis  
- **Green**: Y-axis  
- **Blue**: Z-axis  

---

##  Regression Metrics Included

The project computes the following regression evaluation metrics (via `src/metrics.py`):
- **MSE** — Mean Squared Error  
- **RMSE** — Root Mean Squared Error  
- **MAE** — Mean Absolute Error  
- **R²** — Coefficient of Determination  
- **MedAPE** — Median Absolute Percentage Error  
- **MedSPE** — Median Squared Percentage Error  
- **EVS** — Explained Variance Score  

Each trained model logs these metrics to `output/<model_name>_results.txt`.

---

##  Dependencies

Primary libraries used:
- `opencv-python`
- `mediapipe`
- `numpy`
- `scikit-learn`
- `pandas`
- `scipy`
- `matplotlib`
- `jupyter`

You can install them via:
```bash
pip install -r requirements.txt
```

---

##  Acknowledgements

Developed as a **final-year college project** by **Ali Taleshi**.
