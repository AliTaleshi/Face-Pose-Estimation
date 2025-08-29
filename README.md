# Face Pose Estimation 🎯

Head pose estimation using **MediaPipe FaceMesh** for landmark extraction and **Support Vector Regression (SVR)** for predicting head orientation (pitch, yaw, roll).  
This project supports **dataset-based evaluation** and **real-time webcam inference**.

---

## 📂 Project Structure

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
│ └── *.sav / *.txt
│
├── image.jpg # Test image for model performance evaluation
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

> **Note**: The `output/` directory (containing model files, result logs, plots) and the `dataset/AFLW2000/` directory are excluded from this repository for file size reasons.

---

##  Quick Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AliTaleshi/Face-Pose-Estimation.git
   cd Face-Pose-Estimation

2. **Install dependencies**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt

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

---

### 2. Manual Inference on a New Image
You can use `main.py` to predict pose angles for a specified image:
```bash
python main.py --image path/to/your_image.jpg
```

Example usage inside `main.py` (pseudocode):
```python
from src.models import load_model
from src.preprocessing import preprocess_image

model = load_model('output/SVR_model.sav')
features = preprocess_image(image_path)
pitch, yaw, roll = model.predict([features])[0]
print('Predicted angles — Pitch: {:.2f}, Yaw: {:.2f}, Roll: {:.2f}'.format(pitch, yaw, roll))
```

---

### 3. Real-Time Webcam Inference
For live head pose estimation with axis overlay:
```bash
python src/webcam_inference.py
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

---

##  Future Enhancements

- Incorporate deep learning models (e.g., CNN, ResNet) for regression  
- Support multi-face tracking and multi-angle estimation in real time  
- Deploy as an interactive web app using Flask or Streamlit  
- Expand to video file input for batch inference  
