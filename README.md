# Finger Vein Classification

This project implements a Single Stream ResNet-18 Classification Model for finger vein recognition. The approach treats the problem as a standard image classification task, pooling 6 versions of each image (from the subject's left/right index, middle, and ring fingers) into a single dataset.

The model relies entirely on **Classification** (Softmax) for verification, taking a single raw image as input to predict the unique finger identity out of 600 possible classes (SDUMLA dataset).

## Features
- **Data Loaders & Transform**: Customizable loaders for handling finger vein images, automatically inferring class mapping based on directory structure.
- **Model**: Pre-trained ResNet-18 acting as a backbone, fine-tuned on finger vein images with a custom 512-dimensional embedding layer.
- **Evaluation**: Calculates Test Loss, Test Accuracy, and generates Precision, Recall, F1-scores, and comprehensive ROC curves containing Equal Error Rate (EER) analysis at optimal Thresholds.
- **Support for Two Datasets**:
  - SDUMLA-HMT Database
  - MMCBNU_6000 Database
- Automated train/test splits (80%-20%) per class.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Finger Vascular pattern based Authentication System"
   ```

2. **Install requirements:**
   Install dependencies via pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Configuration:**
   - Update `BASE_DIR_ROOT` and `WORK_DIR_ROOT` in the notebook to match where the datasets are stored.
   - Adjust `USE_SDUMLA=True/False` to toggle between the SDUMLA and MMCBNU databases within the second code cell of the notebook.

## Usage
The entire workflow is self-contained within the `finger_vein_authentication_resnet18.ipynb` notebook.
- Open the notebook using Jupyter:
  ```bash
  jupyter notebook finger_vein_authentication_resnet18.ipynb
  ```
- Run the cells sequentially to build the dataset, instantiate the ResNet-18 model, train for 10 epochs, and evaluate the final test metrics.

## Logs & Results
During training, multiple files will be created in an experiments folder under `WORK_DIR_ROOT`:
- `training_log.csv`: Epoch-by-epoch tracking of Train/Test Accuracy and Loss.
- `best_classification_model.pth`: Saved model weights of the best performing epoch.
- `final_classification_model.pth`: Saved model weights from the last epoch.
- `roc_curve_full_data.csv`: Output of false acceptance/rejection metrics data.
