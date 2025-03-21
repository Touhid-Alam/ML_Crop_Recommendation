# README: A Precise Machine Learning Driven Approach for Crop Recommendation

## Overview
This repository contains the implementation of a **Machine Learning (ML)** based crop recommendation system designed to assist farmers in selecting the most suitable crops based on soil properties and weather conditions. The system leverages a dataset of **2200 instances** with **8 attributes** (e.g., nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall) to train and evaluate multiple ML models, including **Support Vector Machine (SVM)**, **Logistic Regression (LR)**, **K-Nearest Neighbors (KNN)**, **Neural Network (NN)**, and **AdaBoost**. The **KNN model** achieved the best performance with a **testing accuracy of 99.77%**, demonstrating its effectiveness for precise crop recommendation.

---

## Key Features
- **Multi-Class Classification**: Predicts **22 types of crops** based on soil and weather data.
- **High Accuracy**: The **KNN model** achieves **99.77% testing accuracy**, outperforming other models.
- **Feature Importance Analysis**: Identifies **rainfall**, **nitrogen**, and **humidity** as the most influential factors for crop recommendation.
- **Low Computational Cost**: The **KNN model** has a computational cost of **0.0037 seconds**, making it suitable for real-time applications.
- **Comprehensive Evaluation**: Metrics include **accuracy**, **precision**, **recall**, **F1 score**, **MCC**, **MSE**, and **RMSE**.

---

## Dataset
The dataset used in this study is the **Crop Recommendation Dataset**, sourced from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/croprecommendation-dataset). It contains **2200 instances** with **8 attributes** and **22 crop types**. The attributes include:
- **N**: Ratio of Nitrogen
- **P**: Ratio of Phosphorous
- **K**: Ratio of Potassium
- **Temperature**: Temperature in Celsius
- **Humidity**: Relative humidity in %
- **pH**: pH value of the soil
- **Rainfall**: Rainfall in the area in mm
- **Label**: Target attribute indicating the crop to grow

---

## Methodology
![Fig 1](https://github.com/user-attachments/assets/3905557a-8ff4-4806-a342-f8ddef29041c)

### 1. **Exploratory Data Analysis (EDA)**
- The dataset was analyzed to ensure balanced distribution of crop types and identify relationships between attributes.
- Pair plots and boxplots were used to visualize the data distribution and detect outliers.

### 2. **Data Preprocessing**
- **Min-Max Standardization**: Applied to normalize numeric features.
- **Label Encoding**: Converted the target attribute into categorical data.

### 3. **Model Selection**
- **Support Vector Machine (SVM)**: Effective for high-dimensional spaces.
- **Logistic Regression (LR)**: Simple and efficient for multi-class classification.
- **K-Nearest Neighbors (KNN)**: Captures complex decision boundaries.
- **Neural Network (MLPClassifier)**: Handles non-linear relationships.
- **AdaBoost**: Enhances weak classifiers for improved performance.

### 4. **Hyperparameter Tuning**
- **GridSearchCV**: Used to optimize hyperparameters for each model.
- **Tuned Hyperparameters**:
  - SVM: `C`, `kernel`, `gamma`
  - LR: `solver`, `max_iter`
  - KNN: `n_neighbors`
  - NN: `hidden_layer_sizes`, `activation`, `solver`, `max_iter`
  - AdaBoost: `n_estimators`, `learning_rate`

### 5. **Model Training**
- **Train-Test Split**: 80-20 split for training and testing.
- **Cross-Validation**: 5-fold cross-validation to ensure robust evaluation.
- **Random State**: Fixed to `42` for reproducibility.

### 6. **Evaluation Metrics**
- **Accuracy**, **Precision**, **Recall**, **F1 Score**, **MCC**, **MSE**, and **RMSE** were used to evaluate model performance.

### 7. **Feature Importance**
- **Permutation Feature Importance**: Used to identify the most influential features for crop recommendation.

---

## Results
### Model Performance Comparison
| Model               | Testing Accuracy | Precision | Recall | F1 Score | Computational Cost (Seconds) |
|---------------------|------------------|-----------|--------|----------|------------------------------|
| KNN                 | 99.77%           | 99.78%    | 99.77% | 99.77%   | 0.0037                       |
| SVM                 | 99.32%           | 99.37%    | 99.32% | 99.32%   | 0.3267                       |
| Logistic Regression | 98.64%           | 98.72%    | 98.64% | 98.65%   | 0.0859                       |
| Neural Network      | 98.41%           | 98.48%    | 98.41% | 98.42%   | 2.6768                       |
| AdaBoost            | 27.27%           | 17.05%    | 27.27% | 18.69%   | 0.4585                       |

## Visualizations of the findings
![fig 7](https://github.com/user-attachments/assets/1b567f47-b7e8-42fd-82a7-d81d9b717d15)
![fig 14](https://github.com/user-attachments/assets/708df4af-c498-47ff-84e2-24a3e1310421)
![fig 12](https://github.com/user-attachments/assets/5663caf4-8117-434b-af99-884e9502f3bb)
![fig 11](https://github.com/user-attachments/assets/cbdc9ded-d2ff-4928-858e-cf635b82ddab)
![fig 10](https://github.com/user-attachments/assets/0025d1a2-c428-4361-b9ed-8ea6eed686f9)
![fig 9](https://github.com/user-attachments/assets/416efe49-8ffb-419f-9312-0c12785e014a)
![fig 8](https://github.com/user-attachments/assets/4cfc8102-628f-48ca-a077-6062f45f65b3)
### Key Findings
- **KNN** outperformed all other models with a **testing accuracy of 99.77%** and the **lowest computational cost**.
- **Rainfall
**, **Nitrogen**, and **Humidity** were identified as the most important features for crop recommendation.
- **AdaBoost** underperformed due to its reliance on weak learners and sensitivity to data patterns.

---
