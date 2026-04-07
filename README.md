# Mental Health Stress Prediction System (SPL1)

**Developer:** Sadman Sakib (Roll: 1654)  
**Institution:** Institute of Information Technology (IIT), University of Dhaka  
**Course:** Software Project Lab 1 (SPL1)  
**Supervisor:** Dr. Emon Kumar Dey  

---

##  Project Abstract
This project implements a complete machine learning framework from scratch in **pure Java** to predict the stress levels of university students. Moving beyond simple classification, the system is designed with a **medical-centric goal**: to maximize the detection (Recall) of minority stress categories (Low and High) where intervention is most critical. The project demonstrates the full lifecycle of a data science project, from raw data ingestion to rigorous **250-iteration stability testing** and mathematical feature justification.

---

##  Technical Architecture (From Scratch)
The core constraint of this project was the **prohibition of external ML libraries**. Every component was implemented manually using standard Java features.

### 1. Data Pipeline (`data` package)
*   **Robust ETL:** A custom `DataLoader` that handles complex CSV parsing (commas within quoted fields) and cleans malformed data.
*   **Numerical Encoding:** Transformation of categorical ranges (e.g., "18-22", "2.50 - 2.99") into numerical averages.
*   **Feature Normalization:** Implementation of **Min-Max Scaling** to ensure distance-based (KNN) and gradient-based (LR) models are not biased by feature scale.
*   **Class Balancing:** A custom **Minority Oversampling** algorithm to mitigate the severe majority-class bias (66% Moderate Stress).

### 2. Machine Learning Engine (`models` package)
*   **Decision Tree:** Implemented using recursive partitioning, Gini Impurity, and Information Gain.
*   **Logistic Regression:** Multi-class Softmax Regression using Cross-Entropy Loss and Stochastic Gradient Descent (SGD).
*   **K-Nearest Neighbors (KNN):** An instance-based learner using multi-dimensional Euclidean distance and array-based majority voting.

### 3. Evaluation Framework (`evaluation` package)
*   **Confusion Matrix:** A 3x3 matrix implementation to track Actual vs. Predicted results.
*   **Advanced Metrics:** Automated calculation of **Precision, Recall, and F1-Score** per class, allowing for deep analysis of model fairness.

---

##  Scientific Methodology & Findings

### Data-Driven Feature Selection
The feature set was optimized to the "Final 5" based on **Pearson Correlation Analysis** and **Categorical Distribution Analysis**:
1.  **Anxiety Score** (Strongest Predictor: $r = 0.56$)
2.  **Depression Score** (Strong Predictor: $r = 0.50$)
3.  **Gender** (Visible demographic trend: $r = -0.13$)
4.  **CGPA** (Maintained for non-linear academic context)
5.  **Age** (Standard demographic control)

*Note: University and Department were excluded to prevent geographic bias and ensure the model identifies universal symptoms, not locations.*

### The Hyperparameter Tuning Experiment
The system was evaluated over **250 randomized runs** to find stable averages. We successfully navigated the "Accuracy Paradox," choosing to sacrifice ~10% overall accuracy to achieve high sensitivity (Recall) for students in need.

**Optimal Parameters Discovered:**
*   **Decision Tree:** `maxDepth = 5` (Prevents overfitting while maximizing minority recall).
*   **KNN:** `K = 21` (Stabilizes predictions against local noise).

---

##  Results Summary (Averaged Over 250 Runs)
| Scenario | Overall Accuracy | LOW Stress Recall | HIGH Stress Recall |
| :--- | :--- | :--- | :--- |
| **Raw / Imbalanced** | ~72% | ~5% | ~50% |
| **Optimized / Balanced** | **~60%** | **~63%** | **~77%** |

**Scientific Finding:** Addressing class imbalance via oversampling improved the detection of "Low Stress" students by over **1,200%** (from 5% to 63%).

---

##  Project Structure
```text
src/
├── Main.java                 # Orchestrates the 250-run experiment
├── data/
│   ├── DataPoint.java        # Core data structure (features & label)
│   ├── DataLoader.java       # CSV Parser & Transformation Engine
│   └── Preprocessor.java     # Normalization, Oversampling, & Correlation
├── models/
│   ├── DecisionTree.java     # Recursive Partitioning implementation
│   ├── LogisticRegression.java # Softmax/Gradient Descent implementation
│   └── KNN.java              # Distance-based/Voting implementation
└── evaluation/
    └── ConfusionMatrix.java  # Metrics & Reporting engine
