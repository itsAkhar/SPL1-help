# Decision Tree Classifier - Theoretical Guide

This document provides a complete overview of the theory behind the Decision Tree classification algorithm. The goal of this model is to learn a set of hierarchical rules from data to make predictions.

## Core Concept: The Flowchart Model

A Decision Tree is an intuitive model that works like a flowchart. It learns to ask a series of simple yes/no questions about a subject's features (e.g., "Is CGPA less than 3.0?") to arrive at a final conclusion about its class (e.g., "High Stress").

The learning process involves two main phases:
1.  **Building the Tree:** The algorithm automatically discovers the best questions and structure for the flowchart based on the training data.
2.  **Using the Tree:** A new, unseen subject is passed through the finished flowchart to get a prediction.

---

## 1. How the Tree is Built

The tree is built from the top down using a process called **Recursive Partitioning**. The core challenge at every step is to find the single best question to ask.

### **Part A: Measuring "Purity" with Gini Impurity**

To find the "best" question, we first need a way to numerically measure how "mixed up" or "pure" a group of data is.

-   **Concept:** **Gini Impurity** is a score that measures the impurity of a group.
-   **A Gini score of 0** means the group is **perfectly pure** (e.g., all students in the group have "Low Stress"). This is the ideal state.
-   **A high Gini score** means the group is **very mixed up** with multiple classes.
-   **Formula:** The formula is `1 - Σ (pᵢ)²`, where `pᵢ` is the proportion (percentage) of each class `i` in the group. The algorithm's goal is to create groups with the lowest Gini score possible.

### **Part B: Finding the "Best Question" with Information Gain**

The algorithm determines the best question by checking which one creates the "purest" resulting subgroups.

-   **Concept:** The **best question** is the one that leads to the biggest decrease in impurity from the parent group to the child groups. This decrease in impurity is called **Information Gain**.
-   **Process:** The algorithm simulates asking every possible question. For each question:
    1.  It temporarily splits the data into two subgroups (a "yes" group and a "no" group).
    2.  It calculates the Gini Impurity for each of these new subgroups.
    3.  It calculates a **weighted average Gini** for the split, which represents the overall impurity *after* asking the question.
-   **Decision:** The algorithm chooses the question that resulted in the **lowest weighted average Gini**. This is the question that provides the most clarity, or the highest Information Gain.

### **Part C: The Recursive Building Process**

The entire tree is constructed by repeatedly applying the process above.

-   **Logic:**
    1.  Start with the entire dataset at the top (the **root**).
    2.  Use Information Gain to find the best question to split the data.
    3.  This creates two new branches, each with a smaller, slightly purer subset of the data.
    4.  The algorithm then **repeats the entire process independently on each of these new subgroups**, finding the best *next* question for them.
-   **Stopping Conditions:** This recursive process doesn't continue forever. A branch stops growing and becomes a final answer (a **leaf node**) when:
    -   The group becomes perfectly pure.
    -   The tree reaches a predefined maximum depth (a rule to prevent the model from becoming too complex and "memorizing" the data, a problem known as **overfitting**).
    -   The group becomes too small to be worth splitting further.
-   **Leaf Node Prediction:** The prediction for a leaf node is simply the **most common class (the mode)** of the data points that ended up in that final group.

---

## 2. How the Tree Makes a Prediction

Once the tree is fully built, using it is the easy part.

-   **Process (Traversal):** For a new, unseen student:
    1.  Start at the root node.
    2.  Answer the question stored in that node using the student's feature data.
    3.  Follow the corresponding "yes" or "no" branch to the next node.
    4.  Repeat this process, walking down the tree.
    5.  When you arrive at a leaf node, its stored prediction is the final answer for that student.