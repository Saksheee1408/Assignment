# Custom CNN Image Classification System

This project is an end-to-end image classification system built from scratch using TensorFlow/Keras and FastAPI. 
The system trains a Custom Convolutional Neural Network (CNN) without using any pre-trained models or transfer learning.

## Features & Deliverables
- **Data Preparation**: Automatic downloading, splitting (70% Train, 15% Val, 15% Test), and organization of the CIFAR-10 dataset into folders.
- **Data Preprocessing**: Scaling, normalization, and heavy data augmentation (rotations, flips, shifts, brightness).
- **Custom Architectures**:
  - `Custom_CNN_A`: Standard CNN with Conv, BatchNorm, MaxPooling, Dropout, and Dense layers.
  - `Custom_CNN_B` (Bonus): Deeper CNN architecture to compare performance.
- **Model Training**: Trained using Cross-Entropy Loss, Adam Optimizer, Early Stopping, and Model Checkpointing.
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report, and Training History Plots.
- **Grad-CAM (Bonus)**: Implemented Grad-CAM to visualize model attention on image regions.
- **Prediction System**: A visually stunning FastAPI + HTML/CSS/JS frontend to upload images, view predictions, confidence scores, a Chart.js confidence chart, and a feature to export prediction history to CSV.

## Project Structure
```
.
├── app/                  # Web Application (FastAPI + HTML/JS/CSS)
│   ├── static/           # CSS and JS files
│   ├── templates/        # HTML templates
│   └── app.py            # FastAPI backend
├── dataset/              # CIFAR-10 organized by classes
├── models/               # Saved best models (.h5), plots, and history (.csv)
├── notebooks/            # Jupyter notebooks for EDA and Model Training
├── src/                  # Source code for data prep, modeling, training, and evaluation
│   ├── prepare_dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── create_notebook.py    # Script used to generate the notebook
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset**
   This script will download CIFAR-10 and organize it into train, val, and test splits.
   ```bash
   python src/prepare_dataset.py
   ```

3. **Train the Model**
   Train either Architecture A or B.
   ```bash
   python src/train.py --model A --epochs 30 --batch_size 64
   ```

4. **Evaluate the Model**
   Generates Confusion Matrix, Classification Report, and Accuracy/Loss Plots.
   ```bash
   python src/evaluate.py --model models/best_model_A.h5 --history models/history_A.csv
   ```

5. **Start the Web App**
   Start the FastAPI server.
   ```bash
   python app/app.py
   ```
   Then open `http://localhost:8000` in your web browser.

## Bonus Implementations
- Grad-CAM heatmap generator (`src/evaluate.py`).
- Second deeper architecture for comparison (`src/model.py`).
- Prediction Confidence Bar Chart using Chart.js in the frontend.
- Export Session Prediction History to CSV.

## Evaluation Results
*Please refer to `models/confusion_matrix.png` and `models/training_history.png` after running the evaluation script.*
