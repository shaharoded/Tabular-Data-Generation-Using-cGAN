# Tabular-Data-Generation-Using-cGAN

## Overview

This project aims to generate synthetic tabular data using Conditional Generative Adversarial Networks (cGAN). The approach utilizes both real and synthetic data to train models that can effectively mimic the distributions and relationships of features in the original dataset. The model is evaluated based on its ability to generate data with similar feature distributions and its performance in downstream tasks, such as detection and efficacy when used for training a Random Forest model.

![Model Architecture](./Images/cGAN.png)

Key highlights:
- **Conditional GAN**: The model generates synthetic data conditioned on categorical labels, ensuring that the generated data closely matches the characteristics of the original data.
- **Evaluation Metrics**: The quality of the synthetic data is evaluated using detection (AUC) and efficacy (AUC ratio), comparing real and synthetic data performance.
- **Data Preprocessing**: Features are normalized and one-hot encoded for categorical features to ensure that no unintended order is imposed on categorical data.

## Project Structure

### Key Files
- **`config.py`**: Defines model architecture, training parameters, and data handling settings. Includes optimal configurations from experiments.
- **`dataset.py`**: Prepares the dataset by splitting it into training, validation, and test sets. Handles preprocessing such as one-hot encoding and normalization.
- **`gan.py`**: Implements the GAN and cGAN models for data generation.
- **`Analysis.ipynb`**: A notebook for analyzing the quality of the generated data, including correlation differences and feature distributions.
- **`requirements.txt`**: Lists dependencies for running the project.
- **`report.pdf`**: Detailed documentation (in Hebrew) outlining project decisions and experiments.

---

## Usage

### Dataset

The dataset used in this project is assumed to be in `.arff` format. The following preprocessing steps are applied:
- **One-Hot Encoding**: Categorical features are one-hot encoded to avoid forcing any hierarchy using `LabelEncoder`.
- **Normalization**: Numerical features are normalized to ensure consistent scale across all features.

### Environment Setup

Ensure you have Python installed and follow these steps to set up the environment:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Features

- **GAN and Conditional GAN (cGAN)**: The project utilizes both a Generative Adversarial Network (GAN) and a Conditional Generative Adversarial Network (cGAN) to generate synthetic tabular data. The cGAN is conditioned on class labels, ensuring that the generated data matches the feature distribution of each class in the original dataset.
  
- **Data Preprocessing**: 
  - One-hot encoding for categorical features to avoid any ordering in the categorical data.
  - Normalization of numerical features to standardize their range and prevent any scale issues.

- **Detection Metric**: A Random Forest model is trained on a combined dataset consisting of both real and synthetic data. The model is evaluated on its ability to detect whether a given data point is real or synthetic, with a low AUC indicating that the synthetic data is close to the real data.

- **Efficacy Metric**: The model evaluates whether synthetic data can replace real data in training a model. The efficacy is measured as the ratio of the AUC score of a Random Forest model trained on synthetic data to the AUC score of a model trained on real data.

- **Evaluation Visualizations**: 
  - Heatmaps of correlation differences between real and synthetic data.
  - Histograms and feature distributions comparing real and synthetic datasets.
  
- **Random Forest Training**: Multiple models are trained and evaluated using K-fold cross-validation to ensure robust results across different test sets.

---

## Results

The current implementation showed bad results, sufferring from difficulties in handeling the categorical data (one-hot encoded) properly. Improved implementation for this use case was not yet designed. Attached are the current results:

### Detection Metric

The detection metric evaluates how similar the synthetic data is to the real data. A low AUC score in this case is desirable, indicating that the model cannot distinguish between real and synthetic data. A high similarity between the real and synthetic data means that the model is unable to tell the difference between the two.

```
Average AUC for detection (cGAN): 1.0000
```

### Efficacy Metric

The efficacy metric evaluates whether synthetic data can serve as a useful substitute for real data in training models. The efficacy is computed as the ratio of the AUC score for synthetic data to the AUC score for real data, with 1 being the ideal score. A higher ratio suggests that the synthetic data is of high quality and can effectively replace real data.

```
Average AUC for efficacy (cGAN): 0.63
```

### Feature Distribution Comparison

One of the most critical evaluations of the generated data is how well the feature distributions of real and synthetic datasets match. Below is a comparison of the feature distributions, highlighting the effectiveness of the cGAN in replicating the real data distribution.

![Feature Distribution](./Images/features_analysis_cGAN.png)

The plots show histograms of features from both real and synthetic datasets. Ideally, the synthetic data's feature distributions should closely resemble those of the real data.

### Correlation Comparison

Another critical evaluation metric is the correlation between features. The heatmap below illustrates the difference in correlations between the real and synthetic data, highlighting any discrepancies in feature relationships.

![Correlation Heatmap](./Images/cGAN_corr.png)

The closer the correlation difference is to zero, the better the synthetic data mimics the real data in terms of feature interrelationships.

---

## GitHub
### Initiating Git Repository
To clone this repository locally:
```
git init
git remote add origin https://github.com/shaharoded/Tabular-Data-Generation-Using-cGAN.git
git status
```
Alternatively, you can initialize a local repository:
```
git clone https://github.com/shaharoded/Tabular-Data-Generation-Using-cGAN.git
```

### Git Updates
To publish updates:
```
git add .
git commit -m "commit message"
git branch -M main
git push -f origin main
```