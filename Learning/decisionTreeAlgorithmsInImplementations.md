# Project Algorithms Guide: Decision Tree

This document details the core algorithms and data structures used in the from-scratch implementation of the Decision Tree classifier for the Mental Health Stress Prediction project.

## 1. The Decision Tree Model

A Decision Tree is a supervised machine learning model that learns a hierarchical set of rules from data. It creates a flowchart-like structure to make predictions.

-   **`Node.java`**: Represents a single "box" in the flowchart. A node can be one of two types:
    -   **Decision Node**: An internal node that contains a question (`SplitCondition`) and references to two child nodes (a "yes" branch and a "no" branch).
    -   **Leaf Node**: A terminal node that contains a final prediction (the class label) and has no children.

-   **`SplitCondition.java`**: A simple data structure that represents a single yes/no question (e.g., "Is CGPA <= 3.0?"). It holds the index of the feature being questioned and the value it is being compared against.

## 2. Core Training Algorithm: Recursive Partitioning

The tree is built using a greedy, top-down algorithm called **Recursive Partitioning**. The main logic resides in the `buildTree` method of the `DecisionTree.java` class.

-   **Concept:** The algorithm recursively splits the dataset into smaller, purer subsets.
-   **Process:**
    1.  Start with the entire training dataset at the root.
    2.  Find the single best question that splits the current dataset into two subgroups that are as pure as possible. This is the most computationally intensive step.
    3.  If a stopping condition is not met, create a **Decision Node** with this question and repeat the entire process from Step 1 on each of the two new subgroups.
    4.  If a **stopping condition** is met, create a **Leaf Node** and assign it the majority class of the current subgroup.
-   **Stopping Conditions:** To prevent infinite recursion and overfitting, the algorithm stops when:
    -   A node's data is **perfectly pure**.
    -   The tree's **maximum depth** is reached.
    -   A node's data size is below the **minimum samples required to split**.
    -   No split can be found that improves data purity.

## 3. Key Sub-Algorithm: Gini Impurity & Information Gain

To find the "best" split at each node, the algorithm uses the concept of Information Gain, which is calculated using Gini Impurity. This logic is primarily in the `findBestSplit` method.

-   **Gini Impurity:** A metric used to measure the "impurity" or "mixed-up-ness" of a group of data. A score of 0 indicates perfect purity (all data belongs to one class).
    -   **Formula:** `Gini = 1 - Σ (pᵢ)²`, where `pᵢ` is the proportion of class `i` in the group.

-   **Information Gain:** This measures how much "purity" is gained by splitting the data.
    -   **Process:** The algorithm performs a brute-force search across every feature and every possible split value.
    -   For each potential split, it calculates the **weighted average Gini** of the two resulting child groups.
    -   **Formula:** `Gain = Gini(parent) - WeightedGini(children)`
    -   The split that results in the **highest Information Gain** is chosen as the best one for that node.

## 4. Prediction Algorithm: Tree Traversal

Once the tree is built, making a prediction is a simple and fast process called **traversal**. This logic is in the `predict` and `traverseTree` methods.

-   **Process:** For a new, unseen data point:
    1.  Start at the `root` node of the tree.
    2.  At each **Decision Node**, evaluate the node's `SplitCondition` against the data point's features.
    3.  Follow the "yes" (left) or "no" (right) branch based on the outcome.
    4.  Repeat Step 2 and 3 until a **Leaf Node** is reached.
    5.  The prediction stored in that Leaf Node is the final output of the model.