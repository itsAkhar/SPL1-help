# Softmax Regression (Multi-Class Logistic Regression) - Implementation Guide

This document provides a complete theoretical overview and implementation plan for building a Softmax Regression model from scratch in Java. The goal is to classify data into one of three or more categories.

## Core Concept: The Probability Machine

Unlike Linear Regression which answers "How much?", Logistic Regression answers "Which one?". Its goal is not to predict a numerical value, but to calculate the **probability** of an input belonging to each of several distinct classes. The class with the highest probability is the model's final prediction.

The entire process can be broken down into two phases: **Inference (The Forward Pass)** and **Training (The Backward Pass / Gradient Descent)**.

---

## 1. Inference: The Forward Pass

This is the process of taking a single student's data and making a prediction.

**Input:**
- A single student's feature vector `X` (a `1 x 7` array of doubles).
- The trained Weight matrix `W` (a `7 x 3` matrix).
- The trained Bias vector `b` (a `1 x 3` array).

**Steps:**

### **Step A: Calculate Raw Scores (`Z`)**
First, we calculate a linear score for each class ("Low", "Moderate", "High").

- **Formula:** `Z = (X • W) + b`
- **What it means:** For each class, we multiply every feature in `X` by its corresponding weight in `W` for that class, sum them up, and add the class bias. This gives us three raw scores: `[z_low, z_mod, z_high]`. These scores are unbounded.

### **Step B: Convert Scores to Probabilities (`P`)**
Next, we convert these raw, unbounded scores into a clean probability distribution using the **Softmax function**.

- **Formula:** `P(Class i) = e^(zᵢ) / (Σ e^(zⱼ))`
- **What it means:**
    1.  Take the exponential (`e^z`) of each score to make them all positive and amplify the winner.
    2.  Sum up all the exponentiated scores to get a total (the denominator).
    3.  Divide each individual exponentiated score by the total sum.
- **Output:** A vector of probabilities `P` (e.g., `[0.2, 0.7, 0.1]`) where all values are between 0 and 1, and the entire vector sums to 1.

### **Step C: Make a Prediction**
The final prediction is simply the class with the highest probability.

- **Logic:** Find the index of the maximum value in the probability vector `P`. This index corresponds to the predicted class (0, 1, or 2).

---

## 2. Training: Learning the Weights with Gradient Descent

This is the "learning" process. The goal is to find the optimal values for the Weight matrix `W` and Bias vector `b` by iteratively showing the model our training data and nudging the weights to reduce the error.

**Core Idea:** Think of a "Hiker in the Fog" trying to find the bottom of a valley. The "elevation" is our error, and the "slope" is the gradient. The hiker takes small steps downhill until they reach the lowest point.

**The Loop:** For a set number of iterations (epochs):
1. Loop through each student in the training dataset.
2. For each student, perform the following "gradient step":

### **Step A: Make a Prediction (The Forward Pass)**
- Perform the full inference process described above (`Scores -> Softmax -> Probabilities`) using the **current** `W` and `b`.

### **Step B: Calculate the Error Signal (`E`)**
- This is the difference between what the model predicted and the actual truth.
- **Formula:** `E = P - Y`
- **What it means:**
    - `P` is the vector of predicted probabilities (e.g., `[0.2, 0.7, 0.1]`).
    - `Y` is the **one-hot encoded** true label. If the student's true class is "Moderate" (class 1), then `Y = [0, 1, 0]`.
    - The resulting error vector `E` (e.g., `[0.2, -0.3, 0.1]`) shows which classes were over-predicted (positive values) and under-predicted (negative values).

### **Step C: Calculate the Gradients**
- This step "assigns blame" for the error to each weight and bias. It tells us the "slope" of the loss function.
- **Gradient for Weights (`grad_W`):**
    - **Formula:** `grad_W = X_transposed • E`
    - **What it means:** The dot product of the student's feature vector (transposed into a column) and the error vector. The result is a `7 x 3` matrix, the same shape as `W`.
- **Gradient for Biases (`grad_b`):**
    - **Formula:** `grad_b = E`
    - **What it means:** The gradient for the biases is simply the error vector itself.

### **Step D: Update the Parameters**
- This is the "small step downhill." We adjust `W` and `b` in the opposite direction of the gradient, scaled by a `learning_rate`.
- **Formula:**
    - `W_new = W_old - learning_rate * grad_W`
    - `b_new = b_old - learning_rate * grad_b`

By repeating this process thousands of times, the weights and biases are gradually nudged towards their optimal values, where the average **Cross-Entropy Loss (`-log(P_correct_class)`)** is at its minimum.

## Implementation Plan
1.  **Create the `LogisticRegression.java` class.**
2.  **Initialize:** In the constructor, create the `W` matrix and `b` vector and fill them with small random numbers.
3.  **Implement the `train` method:** This will contain the main loops (epochs and the loop over each student).
4.  **Implement helper methods:**
    - `softmax(double[] z)`: A helper to compute the probabilities.
    - `predict(DataPoint)`: A helper to perform the full forward pass.
5.  **Inside the `train` loop:** Call the helpers and implement the logic for calculating `E`, `grad_W`, `grad_b`, and updating the parameters.