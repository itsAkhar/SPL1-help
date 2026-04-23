# Mental Health Stress Prediction System - Data Pipeline

This document provides an overview of the data pipeline for the Mental Health Stress Prediction project. The purpose of this pipeline is to read the raw CSV dataset, parse it, clean it, and transform it into a structured, numerical format that can be used by machine learning algorithms.

## Core Components

The data pipeline consists of two main classes located in the `src/data/` package.

### 1. `DataPoint.java`

This class is the fundamental data structure of the project. It acts as a container for a **single student's processed information**.

-   **Purpose:** To hold all the relevant information for one student in a clean, numerical format.
-   **Key Fields:**
    -   `private final double[] features`: An array of `double` values representing the student's features. The order of features is fixed and consistent.
    -   `private final int label`: An `int` value representing the student's stress level (0 for Low, 1 for Moderate, 2 for High).
-   **Usage:** The entire dataset is loaded into a `List<DataPoint>`. This list is then used for training and testing the machine learning models.

### 2. `DataLoader.java`

This class is responsible for the entire **ETL (Extract, Transform, Load)** process.

-   **Purpose:** To read the raw `MentalHealth.csv` file and convert each row into a `DataPoint` object.
-   **Main Method:**
    -   `public List<DataPoint> loadData(String filePath)`: This is the primary public method. It takes the file path to the CSV, opens it, and iterates through each line, returning a complete `List<DataPoint>`.

-   **Core Logic:**
    1.  **File Reading:** Uses `BufferedReader` to efficiently read the CSV file line by line.
    2.  **Smart Splitting:** It uses a regular expression (`split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1)`) to correctly split CSV lines, even when some fields (like the university name) contain commas within quotes. This prevents data corruption.
    3.  **Feature Transformation:** It uses a series of private helper methods to convert raw text from the CSV into numerical features.

-   **Helper Methods & Transformation Rules:**
    -   `encodeAge(String)`: Converts age ranges (e.g., "18-22") into their numerical average (e.g., `20.0`).
    -   `encodeGender(String)`: Encodes gender strings ("Female", "Male") into numbers (`0.0`, `1.0`).
    -   `encodeAcademicYear(String)`: Encodes academic years ("First Year", "Second Year", etc.) into zero-based numbers (`0.0`, `1.0`, etc.).
    -   `encodeCgpa(String)`: Converts CGPA ranges (e.g., "2.50 - 2.99") into their numerical average (e.g., `2.745`).
    -   `encodeScholarship(String)`: Encodes "Yes"/"No" strings into binary numbers (`1.0`/`0.0`).
    -   **Direct Parsing:** For features that are already numbers (like `Anxiety Value` and `Depression Value`), it cleans the string by removing quotes and parses them directly into `double` values.
    -   `encodeStressLabel(String)`: Encodes the target label ("Low Stress", "Moderate Stress", "High Perceived Stress") into the integer classes `0`, `1`, and `2`.

-   **Error Handling:** Each line is processed within a `try-catch` block. If a line is malformed or causes a parsing error, it is skipped, and an error message is printed to the console without crashing the entire program.

## Final Feature Vector

The `DataLoader` produces a `DataPoint` for each student with a feature array structured in the following order:

1.  `Age` (encoded)
2.  `Gender` (encoded)
3.  `Academic_Year` (encoded)
4.  `Current_CGPA` (encoded)
5.  `waiver_or_scholarship` (encoded)
6.  `Anxiety Value` (parsed)
7.  `Depression Value` (parsed)

This structured, numerical data is the final output of the pipeline and the required input for the next phase: **Model Training**.