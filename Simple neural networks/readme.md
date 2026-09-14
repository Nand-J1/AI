# Digit Recognizer 🔢

## Overview
This project implements a simple neural network from scratch to classify handwritten digits from the MNIST dataset using NumPy. It demonstrates the fundamentals of forward propagation, backpropagation, and gradient descent.

## Dataset
- Source: Kaggle Digit Recognizer competition (`train.csv`)
- Each image: 28×28 pixels → 784 features
- Labels: Digits 0–9

## Workflow
1. Data Preprocessing: load CSV, normalize pixel values, split into training and development sets.
2. Model Architecture: input layer (784 neurons), hidden layer (10 neurons, ReLU), output layer (10 neurons, Softmax).
3. Training: gradient descent from scratch, accuracy printed every 10 iterations, reaches ~84% after 500 iterations.
4. Prediction & Visualization: test predictions on sample images, display digit images with predicted vs. actual labels.

## Example Training Output
i= 0   accuracy= 0.09  
i= 100 accuracy= 0.58  
i= 300 accuracy= 0.79  
i= 490 accuracy= 0.84  

## Requirements
- Python 3.12+
- NumPy
- Pandas
- Matplotlib

Install dependencies:
pip install numpy pandas matplotlib

## Usage
Run the notebook:
jupyter notebook digit_recognizer.ipynb

## Future Improvements
- Add momentum gradient descent
- Implement Adam optimizer
- Extend to convolutional neural networks (CNNs) for higher accuracy
