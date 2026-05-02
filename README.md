# Mental Health Stress Prediction Using ML
### A From-Scratch Implementation in Java

> **SPL1 Project — Institute of Information Technology (IIT), University of Dhaka**
> **Student:** Sadman Sakib | **Roll:** 1654
> **Supervisor:** Dr. Emon Kumar Dey

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Analysis](#feature-analysis)
- [Project Structure](#project-structure)
- [Pipeline — 4 Acts](#pipeline--4-acts)
- [Model Implementation](#model-implementation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project develops a machine learning system that predicts the **stress levels of university students** based on demographic, academic, and psychological data. Students are classified into three categories:

| Label | PSS Score Range |
|---|---|
| 🟢 Low Stress | 0 – 13 |
| 🟡 Moderate Stress | 14 – 26 |
| 🔴 High Perceived Stress | 27 – 40 |

All ML algorithms are **implemented entirely from scratch in Java** — no external ML libraries are used. The goal is to demonstrate deep algorithmic understanding while solving a real-world mental health detection problem.

---

## Dataset

**Source:** [MHP (Anxiety, Stress, Depression) Dataset of University Students — figshare](https://figshare.com)

| Property | Value |
|---|---|
| Total Records | 2,028 |
| Population | Bangladeshi university students |
| Format | CSV |

### Original Features (10 total)

| Feature | Type | Description |
|---|---|---|
| Age | Numerical (continuous) | Age range encoded as midpoint |
| Gender | Categorical (nominal) | Female / Male / Prefer not to say |
| University | Categorical (nominal) | 15 Bangladeshi universities |
| Department | Categorical (nominal) | 12 department categories |
| Academic Year | Categorical (ordinal) | First / Second / Third / Fourth / Other |
| Current CGPA | Numerical (continuous) | CGPA range encoded as midpoint |
| Waiver/Scholarship | Categorical (binary) | Yes / No |
| **Stress Label** | **Target** | **Low / Moderate / High (PSS score)** |
| Anxiety Value | Numerical (continuous) | GAD-7 score (0–21) |
| Depression Value | Numerical (continuous) | PHQ-9 score (0–27) |

---

## Feature Analysis

### Pearson Correlation
Applied to all **continuous/numerical features** to measure linear association with the stress label.

![Pearson Correlation Output](images/pearson_correlation.png)

| Feature | r value | Relationship |
|---|---|---|
| Age | -0.05 | NEGLIGIBLE |
| CGPA | -0.06 | NEGLIGIBLE |
| Anxiety Score | 0.56 | MODERATE |
| Depression Score | 0.50 | MODERATE |

> Pearson correlation is applicable only to numerical/continuous variables. Gender, Academic Year, and Waiver are categorical — analyzed separately using group distribution tables.

---

### Group Distribution Analysis

**Gender Distribution**

![Gender Distribution Table](images/gender_distribution.png)

Gender is retained for **demographic context** despite weak correlation, as stress patterns differ noticeably across gender groups.

---

**Academic Year Distribution**

![Academic Year Distribution Table](images/academic_year_distribution.png)

---

**Waiver / Scholarship Distribution**

![Waiver Distribution Table](images/waiver_distribution.png)

Academic Year and Waiver are **excluded** due to negligible correlation and near-uniform stress distribution across groups — they add noise without predictive value.

---

### Stress Distribution by University & Department

![University Distribution Chart](images/university_distribution.png)

![Department Distribution Chart](images/department_distribution.png)

University and Department are excluded to **avoid institutional bias** and maintain model generalizability.

---

### Final Selected Features (5)

| Feature | Reason for Inclusion |
|---|---|
| Age | Continuous — included after correlation analysis |
| Gender | Nominal — included for demographic context |
| CGPA | Continuous — included after correlation analysis |
| Anxiety Value | Strong predictor (r = 0.56) |
| Depression Value | Strong predictor (r = 0.50) |

---

### Class Imbalance

The dataset is significantly imbalanced, causing models to bias results toward the majority class (Moderate Stress).

| Class | Count | Percentage |
|---|---|---|
| Moderate Stress | 1,348 | 66.5% |
| High Perceived Stress | 565 | 27.9% |
| Low Stress | 115 | 5.7% |

This imbalance is addressed in **Act 3** using oversampling.

---

## Project Structure

```
SPL1-Mental-Health-Stress-Prediction/
│
├── src/
│   ├── Main.java                          # Entry point — interactive menu
│   │
│   ├── data/
│   │   ├── DataLoader.java                # CSV parsing & feature encoding
│   │   ├── DataPoint.java                 # Data model (features + label + metadata)
│   │   └── Preprocessor.java             # Normalization, oversampling, K-Fold split
│   │
│   ├── models/
│   │   ├── logisticRegression/
│   │   │   └── LogisticRegression.java    # Softmax + SGD from scratch
│   │   ├── knn/
│   │   │   └── KNN.java                   # K-Nearest Neighbors from scratch
│   │   └── decisionTree/
│   │       ├── DecisionTree.java          # Gini impurity + recursive split
│   │       ├── Node.java                  # Tree node structure
│   │       ├── BestSplitResult.java       # Stores best split candidate
│   │       └── SplitCondition.java        # Split condition definition
│   │
│   └── evaluation/
│       └── ConfusionMatrix.java           # Accuracy, Precision, Recall, F1
│
├── MentalHealth.csv                       # Dataset (place in root directory)
├── images/                                # Charts and screenshots for README
└── README.md
```

---

## Pipeline — 4 Acts

The evaluation pipeline is structured into 4 progressive acts, each building on the previous to address identified weaknesses.

```
Raw Data
   │
   ▼
ACT 1: Imbalanced (Baseline)
   │  → Reveals bias toward Moderate class
   ▼
ACT 2: Normalized (Min-Max)
   │  → Improves distance-based models (KNN, LR)
   ▼
ACT 3: Balanced + Normalized (Oversampling)
   │  → Dramatically improves Low & High stress recall
   ▼
ACT 4: Optimized (Tuned Hyperparameters)
   └  → Best overall performance (maxDepth=5, k=21)
```

All acts use **10-Fold Cross-Validation** to ensure every data point is used for both training and testing.

---

## Model Implementation

All three models are implemented **from scratch in Java** with no external ML libraries.

### 1. Decision Tree
- **Algorithm:** Recursive binary splitting using Gini Impurity
- **Initial depth:** 10 | **Optimized depth:** 5
- **Min samples per split:** 2
- **Why chosen:** Interpretable decision paths, handles non-linear relationships, no normalization required

### 2. Logistic Regression
- **Algorithm:** Softmax regression with Stochastic Gradient Descent (SGD)
- **Learning rate:** 0.01 | **Epochs:** 500
- **Why chosen:** Proven baseline linear classifier, probabilistic output, fast training for repeated CV

### 3. K-Nearest Neighbors (KNN)
- **Algorithm:** Euclidean distance + majority vote
- **Initial k:** 5 | **Optimized k:** 21
- **Why chosen:** Non-parametric, instance-based learning, sensitive to normalization (good for demonstrating preprocessing impact)

---

## Hyperparameter Tuning

Tuning was performed on the **Act 3 pipeline** (balanced + normalized) to find optimal parameters that maximize minority class recall.

### KNN — k Parameter Tuning

![KNN Tuning Chart](images/knn_tuning.png)

**Chosen k = 21** — LOW recall plateaus and HIGH recall is maximized without accuracy collapsing.

---

### Decision Tree — maxDepth Tuning

![Decision Tree Tuning Chart](images/dt_tuning.png)

**Chosen maxDepth = 5** — LOW recall (~59%) stays good while HIGH recall (~75%) remains strong before both degrade at deeper depths.

---

## Results

All results are averaged over **50 repeats × 10-Fold Cross-Validation**.

### Overall Accuracy — Act 1 → Act 4

![Overall Accuracy Chart](images/overall_accuracy.png)

| Act | Decision Tree | Logistic Regression | KNN |
|---|---|---|---|
| Act 1 (Imbalanced) | 69.8% | 62.7% | 72.4% |
| Act 2 (Normalized) | 69.8% | 74.3% | 72.0% |
| Act 3 (Balanced+Norm) | 60.2% | 55.6% | 59.1% |
| Act 4 (Optimized) | 61.0% | 55.6% | 53.6% |

---

### Act 1: Imbalanced Data (Baseline)

![Act 1 Chart](images/act1_chart.png)

> High accuracy but poor Low and High stress recall — model is biased toward Moderate class due to imbalance.

---

### Act 2: Normalized Data

![Act 2 Chart](images/act2_chart.png)

> Min-max normalization applied. LR accuracy jumps to 74.30% but Low-stress recall collapses to 2.38% for LR.

---

### Act 3: Balanced + Normalized

![Act 3 Chart](images/act3_chart.png)

> Oversampling applied. Overall accuracy drops slightly but Low and High stress recall improves dramatically across all models.

---

### Recall — Low Stress Class (Act 1 → 4)

![Low Stress Recall Chart](images/low_recall.png)

| Act | Decision Tree | Logistic Regression | KNN |
|---|---|---|---|
| Act 1 | 13.8% | 18.2% | 5.7% |
| Act 2 | 13.7% | 2.4% | 6.3% |
| Act 3 | 44.0% | 72.8% | 37.9% |
| Act 4 | 60.4% | 72.8% | 62.1% |

---

### Recall — High Stress Class (Act 1 → 4)

![High Stress Recall Chart](images/high_recall.png)

| Act | Decision Tree | Logistic Regression | KNN |
|---|---|---|---|
| Act 1 | 56.0% | 54.3% | 55.8% |
| Act 2 | 56.0% | 56.9% | 54.9% |
| Act 3 | 68.3% | 78.9% | 71.3% |
| Act 4 | 75.4% | 78.9% | 76.8% |

---

### Best Model: Logistic Regression (Act 4)

| Metric | Value |
|---|---|
| Overall Accuracy | 55.6% |
| Low Stress Recall | **72.8%** |
| High Stress Recall | **78.9%** |
| Pipeline | Balanced + Normalized + Tuned |

> In a mental health prediction context, **recall for minority classes (Low and High stress) is more critical than overall accuracy**. Missing a high-stress student is a far worse outcome than a false alarm.

---

## How to Run

### Prerequisites
- Java JDK 8 or higher
- IntelliJ IDEA (recommended) or any Java IDE
- `MentalHealth.csv` placed in the **root/working directory**

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/SPL1-Mental-Health-Stress-Prediction.git
cd SPL1-Mental-Health-Stress-Prediction
```

**2. Open in IntelliJ IDEA**
```
File → Open → Select project folder
Mark src/ as Sources Root
```

**3. Place the dataset**
```
MentalHealth.csv  ← place in root directory (same level as src/)
```

**4. Run Main.java**
```
Right-click Main.java → Run 'Main.main()'
```

**5. Use the interactive menu**
```
==========================================================
     Mental Health Stress Prediction System - IIT, DU
==========================================================

----------------------------------------------------------
Please select an option:
  1 -> Feature Analysis
  2 -> Model Evaluation (10-Fold CV)
  3 -> Predict Stress for a New Student
  0 -> Exit
Your choice:
```

### Menu Options

| Option | Description |
|---|---|
| `1` | Feature Analysis — Pearson correlation + group distribution tables |
| `2` | Model Evaluation — Runs full 10-Fold CV across all 4 Acts |
| `3` | Predict — Enter a new student's data and get stress prediction from 3 models with majority vote |
| `0` | Exit |

### Changing Number of Repeats
In `Main.java`, change the constant:
```java
private static final int NUM_REPEATS = 1;   // Change to 50 for full analysis
```

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Java | Primary programming language |
| IntelliJ IDEA | Development environment |
| Git & GitHub | Version control |
| CSV | Dataset format |
| Manual Math | All ML algorithms implemented from scratch |

---

## Acknowledgements

- **Dataset:** MHP (Anxiety, Stress, Depression) Dataset — figshare
- **Supervisor:** Dr. Emon Kumar Dey, IIT, University of Dhaka
- **Institution:** Institute of Information Technology (IIT), University of Dhaka

---

*SPL1 Project — 2025*
