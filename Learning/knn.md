# K-Nearest Neighbors (KNN) Classifier - Guide

This document provides the theoretical overview and implementation plan for the K-Nearest Neighbors (KNN) classifier, built from scratch in Java.

## Core Concept: "You Are Like Your Neighbors"

KNN is one of the simplest and most intuitive machine learning algorithms. It is a **lazy learner**, meaning it does not have a distinct training phase. Instead, it memorizes the entire training dataset.

The core idea is: **To classify a new, unknown data point, find the "K" most similar points to it in the known dataset and take a majority vote of their classes.**

-   **"K"**: A user-defined integer that specifies how many neighbors to look at (e.g., 3-NN, 5-NN). This is the model's main hyperparameter.
-   **"Nearest" / "Similar"**: Similarity is determined by a **distance metric**. The most common is **Euclidean Distance**.

---

## 1. The Algorithm

### **Part A: The "Training" Phase**

The training phase is trivial:
1.  Load the entire training dataset.
2.  Store it in memory.

That's it. No calculations are performed.

### **Part B: The Prediction Phase**

This is where all the computation happens. To predict the class of a new data point:

1.  **Calculate Distances:** Iterate through **every single data point** in the stored training set and calculate the distance between it and the new data point.
    -   **Euclidean Distance Formula:** For two points, A and B, with *n* features:
        `Distance = √[ (A₁-B₁)² + (A₂-B₂)² + ... + (Aₙ-Bₙ)² ]`

2.  **Find the K-Nearest Neighbors:**
    -   Store all the calculated distances along with the corresponding labels from the training data.
    -   Sort this list of neighbors in ascending order based on their distance.
    -   Select the first `K` elements from the sorted list. These are the "K-Nearest Neighbors."

3.  **Take a Majority Vote:**
    -   Look at the class labels of these `K` neighbors.
    -   Count the occurrences of each class (e.g., 3 votes for "High", 1 for "Moderate", 1 for "Low").
    -   The class with the highest vote count is the final prediction for the new data point.

---

## 2. Implementation Plan in Java

The implementation is encapsulated within the `KNN.java` class.

-   **Fields:**
    -   `int k`: Stores the number of neighbors to use.
    -   `List<DataPoint> trainingData`: The "memory" of the model where the entire dataset is stored.

-   **Key Methods:**
    -   `train(List<DataPoint> data)`: A simple method that assigns the input data to the `trainingData` field.
    -   `predict(DataPoint dp)`: The main method that orchestrates the entire prediction process (Calculate Distances -> Find Neighbors -> Majority Vote).
    -   `euclideanDistance(double[] featuresA, double[] featuresB)`: A private helper method that implements the mathematical distance formula.

-   **Helper Structures:**
    -   A small private inner class (e.g., `Neighbor`) is useful to hold pairs of `(distance, label)` to make sorting easier.