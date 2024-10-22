# Handwritten Digits Classification Project

This project aims to classify handwritten digits using various machine learning algorithms and neural networks. The dataset used in this project is the **MNIST dataset**, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9). The primary goal of this project is to compare the performance of different machine learning models and optimize their performance using grid search for hyperparameter tuning.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Algorithms Used](#algorithms-used)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction
This project demonstrates the use of various machine learning techniques to classify handwritten digits from the MNIST dataset. We experiment with the following algorithms:
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Logistic Regression
- Random Forest Classifier

We also implement hyperparameter tuning using **GridSearchCV** to optimize the models' performance. Additionally, we normalize the data and compare the performance of models with and without normalization.

## Project Structure

The project is organized as follows:
. ├── data/ # Contains the MNIST dataset ├── notebooks/ # Jupyter notebooks with code and experiments ├── models/ # Saved models and grid search results ├── src/ # Source code for the models and scripts ├── README.md # Project documentation

## Algorithms Used

### 1. K-Nearest Neighbors (KNN)
KNN is a non-parametric algorithm used for classification. It works by finding the `k` closest neighbors in the feature space and making predictions based on the majority class.

### 2. Decision Tree Classifier
Decision Trees are a tree-based algorithm where data is split based on feature values to create a tree-like structure to make predictions.

### 3. Logistic Regression
A parametric algorithm that uses a sigmoid function to model the probability of binary classes (extended for multi-class problems using softmax).

### 4. Random Forest Classifier
An ensemble learning method that combines multiple decision trees (bagging) to improve the robustness and accuracy of the model.

## Hyperparameter Tuning

We use **GridSearchCV** to find the best set of hyperparameters for KNN and Random Forest classifiers. The following parameters are tuned:

- **KNN**:
  - `n_neighbors`: [3, 4]
  - `weights`: ['uniform', 'distance']
  - `algorithm`: ['auto', 'brute']

- **Random Forest**:
  - `n_estimators`: [300, 500]
  - `max_features`: ['sqrt', 'log2']
  - `class_weight`: ['balanced', 'balanced_subsample']

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Scikit-learn
- Matplotlib
- Pandas
- Jupyter

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/handwritten-digits-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd handwritten-digits-classification
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
