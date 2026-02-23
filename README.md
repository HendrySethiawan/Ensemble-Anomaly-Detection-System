# Master's Thesis: Ensemble Anomaly Detection System

This repository contains the implementation of a Master's thesis project focused on unsupervised anomaly detection. The project utilizes an ensemble approach combining **Isolation Forest** and **Local Outlier Factor (LOF)** to identify outliers in high-dimensional datasets.

## 📋 Overview

The project includes two main files for demonstration and analysis:
1.  **`Anomaly_Detection_Demo.py`**: A standalone Python script for automated execution.
2.  **`Anomaly_Detection_Demo.ipynb`**: An interactive Jupyter Notebook for detailed exploration and visualization.

The script performs the following steps:
1.  **Data Preprocessing:** Scales the data using `RobustScaler` to minimize the influence of outliers during the training phase.
2.  **Model Training:** Trains two distinct unsupervised learning models (Isolation Forest and LOF).
3.  **Ensemble Prediction:** Combines the predictions of both models to improve detection accuracy and reduce false positives.
4.  **Visualization:** Uses Principal Component Analysis (PCA) to visualize the anomalies in a 2D space.
5.  **Analysis:** Performs statistical profiling, sensitivity analysis, and synthetic anomaly injection tests.

## 🛠️ Tech Stack

*   **Python:** Core programming language.
*   **Pandas:** Data manipulation and analysis.
*   **Scikit-Learn:** Machine learning algorithms (Isolation Forest, LOF, PCA, RobustScaler).
*   **Matplotlib & Seaborn:** Data visualization.

## 📦 Installation

### 1. Create `requirements.txt`
Create a file named `requirements.txt` in your project folder and add the following lines:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Option A: Run the Python Script
1.  Ensure you have a CSV file named `sample1.csv` in the same directory (or update the file path in the script).
2.  Run the script:

```bash
python Anomaly_Detection_Demo.py
```

### Option B: Run the Jupyter Notebook
1.  Open `Anomaly_Detection_Demo.ipynb` in Jupyter Notebook or JupyterLab.
2.  Run the cells sequentially to see the data processing, model training, and visualizations.

## 🔬 Methodology

### 1. Isolation Forest
An ensemble method that isolates anomalies by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

### 2. Local Outlier Factor (LOF)
A density-based algorithm that calculates the local density deviation of a given data point with respect to its neighbors. It is particularly effective for detecting local anomalies.

### 3. Ensemble Strategy
The final prediction is determined by taking the average of the binary predictions from both models. If the average score is greater than or equal to 0.5, the data point is classified as an anomaly.

### 4. Sensitivity Analysis
The code includes a sensitivity analysis loop that tests the model's performance under different contamination levels (e.g., 0.05, 0.1, 0.15) to ensure robustness.

## 📄 License

This code is provided for academic and research purposes.
