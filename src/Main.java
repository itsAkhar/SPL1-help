import data.DataLoader;
import data.DataPoint;
import data.Preprocessor;
import evaluation.ConfusionMatrix;
import models.decisionTree.DecisionTree;
import models.knn.KNN;
import models.logisticRegression.LogisticRegression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class Main {

    // =========================================================================
    // CONSTANTS
    // =========================================================================
    private static final int NUM_CLASSES  = 3;
    private static final int K_FOLDS      = 10;
    private static final int NUM_REPEATS  = 1;   // Change manually as needed

    // Act 4 optimal hyperparameters
    private static final int OPT_MAX_DEPTH = 5;
    private static final int OPT_K         = 21;

    // Stress label names
    private static final String[] STRESS_LABELS = {"Low Stress", "Moderate Stress", "High Perceived Stress"};

    // Feature names
    private static final String[] FEATURE_NAMES = {"Age", "Gender", "CGPA", "Anxiety Score", "Depression Score"};

    // =========================================================================
    // MAIN
    // =========================================================================
    public static void main(String[] args) {
        System.out.println("==========================================================");
        System.out.println("     Mental Health Stress Prediction System - IIT, DU     ");
        System.out.println("==========================================================");

        DataLoader loader = new DataLoader();
        String filePath = "MentalHealth.csv";
        List<DataPoint> allData = loader.loadData(filePath);

        if (allData.isEmpty()) {
            System.err.println("ERROR: No data loaded. Check the file path and CSV format.");
            return;
        }

        int numFeatures = allData.get(0).getFeatureCount();
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("\n----------------------------------------------------------");
            System.out.println("Please select an option:");
            System.out.println("  1 -> Feature Analysis");
            System.out.println("  2 -> Model Evaluation (10-Fold CV)");
            System.out.println("  3 -> Predict Stress for a New Student");
            System.out.println("  0 -> Exit");
            System.out.print("Your choice: ");

            String input = scanner.nextLine().trim();

            switch (input) {
                case "1":
                    runFeatureAnalysis(allData);
                    break;
                case "2":
                    runEvaluation(allData, numFeatures);
                    break;
                case "3":
                    runPrediction(allData, numFeatures, scanner);
                    break;
                case "0":
                    System.out.println("\nExiting. Goodbye!");
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid choice. Please enter 1, 2, 3, or 0.");
            }
        }
    }

    // =========================================================================
    // OPTION 1: FEATURE ANALYSIS
    // =========================================================================
    private static void runFeatureAnalysis(List<DataPoint> allData) {
        System.out.println("\n\n==========================================================");
        System.out.println("              OPTION 1: FEATURE ANALYSIS                 ");
        System.out.println("==========================================================");

        // ------------------------------------------------------------------
        // PART A: Pearson Correlation — Age(0), CGPA(2), Anxiety(3), Depression(4)
        // Gender(1) is nominal — Pearson not applicable
        // ------------------------------------------------------------------
        System.out.println("\n--- Pearson Correlation with Stress Label ---");
        System.out.println("(Applicable only to numerical/continuous features)");
        System.out.println("----------------------------------------------------------");
        System.out.printf("%-20s | %-10s | %-15s%n", "Feature", "r value", "Relationship");
        System.out.println("----------------------------------------------------------");

        int[] pearsonIndices = {0, 2, 3, 4}; // Age, CGPA, Anxiety, Depression
        for (int idx : pearsonIndices) {
            double r = Preprocessor.calculateCorrelation(allData, idx);
            String strength = getCorrelationStrength(r);
            System.out.printf("%-20s | r = %-7.2f | %-15s%n", FEATURE_NAMES[idx], r, strength);
        }
        System.out.println("----------------------------------------------------------");
        System.out.println("NOTE: Gender is a nominal categorical variable.");
        System.out.println("      Pearson correlation is not applicable. See group table below.");

        // ------------------------------------------------------------------
        // PART B: Group Distribution Table — Gender
        // Encoding: 0=Female, 1=Male, 2=Prefer not to say
        // ------------------------------------------------------------------
        System.out.println("\n--- Stress Label Distribution by Gender ---");
        String[] genderNames = {"Female", "Male", "Prefer not to say"};
        printGroupDistributionTable(allData, 1, genderNames);

        // ------------------------------------------------------------------
        // PART C: Group Distribution Table — Academic Year
        // NOTE: Academic Year is encoded in DataLoader but NOT added to the
        // features[] array. It cannot be accessed from DataPoint objects.
        // ------------------------------------------------------------------
        System.out.println("\n--- Stress Label Distribution by Academic Year ---");
        String[] academicYearNames = {"First Year", "Second Year", "Third Year", "Fourth Year", "Other"};
        printMetadataGroupDistributionTable(allData, "academicYear", academicYearNames);

        // ------------------------------------------------------------------
        // PART D: Group Distribution Table — Waiver / Scholarship
        // NOTE: Same situation — encoded but not in features array.
        // ------------------------------------------------------------------
        System.out.println("\n--- Stress Label Distribution by Waiver / Scholarship ---");
        String[] waiverNames = {"No Waiver", "Has Waiver"};
        printMetadataGroupDistributionTable(allData, "waiver", waiverNames);
    }

    /**
     * Prints a group distribution table showing count and % of each stress label per group.
     *
     * @param allData    The full dataset.
     * @param featureIdx The index of the grouping feature in the features array.
     * @param groupNames Display names for each group (index = encoded value).
     */
    private static void printGroupDistributionTable(List<DataPoint> allData,
                                                    int featureIdx,
                                                    String[] groupNames) {
        int numGroups = groupNames.length;
        int[][] counts = new int[numGroups][NUM_CLASSES];
        int[] groupTotals = new int[numGroups];

        for (DataPoint dp : allData) {
            int group = (int) dp.getFeature(featureIdx);
            int label = dp.getLabel();
            if (group >= 0 && group < numGroups && label >= 0 && label < NUM_CLASSES) {
                counts[group][label]++;
                groupTotals[group]++;
            }
        }

        System.out.println("------------------------------------------------------------------");
        System.out.printf("%-22s | %-5s | %-16s | %-16s | %-22s%n",
                "Group", "N", "Low Stress", "Moderate Stress", "High Perceived Stress");
        System.out.println("------------------------------------------------------------------");
        for (int g = 0; g < numGroups; g++) {
            int total = groupTotals[g];
            if (total == 0) continue;
            System.out.printf("%-22s | %-5d | %-16s | %-16s | %-22s%n",
                    groupNames[g], total,
                    formatCount(counts[g][0], total),
                    formatCount(counts[g][1], total),
                    formatCount(counts[g][2], total));
        }
        System.out.println("------------------------------------------------------------------");
    }

    /**
     * Prints a group distribution table for metadata fields (academicYear or waiver)
     * stored directly on DataPoint — NOT in the features[] array.
     *
     * @param allData    The full dataset.
     * @param field      Which metadata field to group by: "academicYear" or "waiver".
     * @param groupNames Display names for each group (index = encoded value).
     */
    private static void printMetadataGroupDistributionTable(List<DataPoint> allData,
                                                            String field,
                                                            String[] groupNames) {
        int numGroups = groupNames.length;
        int[][] counts = new int[numGroups][NUM_CLASSES];
        int[] groupTotals = new int[numGroups];

        for (DataPoint dp : allData) {
            int group;
            if (field.equals("academicYear")) {
                group = dp.getAcademicYear();
            } else { // "waiver"
                group = dp.getWaiver();
            }
            int label = dp.getLabel();
            if (group >= 0 && group < numGroups && label >= 0 && label < NUM_CLASSES) {
                counts[group][label]++;
                groupTotals[group]++;
            }
        }

        System.out.println("------------------------------------------------------------------");
        System.out.printf("%-22s | %-5s | %-16s | %-16s | %-22s%n",
                "Group", "N", "Low Stress", "Moderate Stress", "High Perceived Stress");
        System.out.println("------------------------------------------------------------------");
        for (int g = 0; g < numGroups; g++) {
            int total = groupTotals[g];
            if (total == 0) continue;
            System.out.printf("%-22s | %-5d | %-16s | %-16s | %-22s%n",
                    groupNames[g], total,
                    formatCount(counts[g][0], total),
                    formatCount(counts[g][1], total),
                    formatCount(counts[g][2], total));
        }
        System.out.println("------------------------------------------------------------------");
    }

    private static String formatCount(int count, int total) {
        double pct = (total > 0) ? (100.0 * count / total) : 0.0;
        return count + " (" + String.format("%.1f", pct) + "%)";
    }

    private static String getCorrelationStrength(double r) {
        double abs = Math.abs(r);
        if (abs >= 0.7) return "STRONG";
        if (abs >= 0.4) return "MODERATE";
        if (abs >= 0.1) return "WEAK";
        return "NEGLIGIBLE";
    }

    // =========================================================================
    // OPTION 2: EVALUATION (logic unchanged from original Main)
    // =========================================================================
    private static void runEvaluation(List<DataPoint> allData, int numFeatures) {
        System.out.println("\n\n==========================================================");
        System.out.println("   OPTION 2: MODEL EVALUATION - 10-Fold Cross-Validation  ");
        System.out.println("==========================================================");

        Collections.shuffle(allData);

        ConfusionMatrix dtAct1  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix lrAct1  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix knnAct1 = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix dtAct2  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix lrAct2  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix knnAct2 = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix dtAct3  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix lrAct3  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix knnAct3 = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix dtAct4  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix lrAct4  = new ConfusionMatrix(NUM_CLASSES);
        ConfusionMatrix knnAct4 = new ConfusionMatrix(NUM_CLASSES);

        System.out.println("Running " + NUM_REPEATS + " repeats of " + K_FOLDS + "-Fold CV.");
        System.out.println("Total models to train per Act: " + (NUM_REPEATS * K_FOLDS));

        for (int repeat = 1; repeat <= NUM_REPEATS; repeat++) {
            System.out.println("\n--- Starting Repeat " + repeat + " of " + NUM_REPEATS + " ---");
            Collections.shuffle(allData);

            for (int k = 0; k < K_FOLDS; k++) {
                System.out.print("  Processing Fold " + (k + 1) + "/" + K_FOLDS + "...\r");

                List<List<DataPoint>> splitData = Preprocessor.getKFoldSplit(allData, K_FOLDS, k);
                List<DataPoint> rawTrain = splitData.get(0);
                List<DataPoint> rawTest  = splitData.get(1);

                // ACT 1: Imbalanced & Un-normalized
                runModelsAndAddPredictions(
                        deepCopy(rawTrain), deepCopy(rawTest),
                        numFeatures, NUM_CLASSES,
                        dtAct1, lrAct1, knnAct1, 10, 5);

                // ACT 2: Imbalanced & Normalized
                List<DataPoint> trainNorm = deepCopy(rawTrain);
                List<DataPoint> testNorm  = deepCopy(rawTest);
                Preprocessor.normalize(trainNorm, testNorm);
                runModelsAndAddPredictions(
                        trainNorm, testNorm,
                        numFeatures, NUM_CLASSES,
                        dtAct2, lrAct2, knnAct2, 10, 5);

                // ACT 3: Balanced & Normalized (Default Params)
                List<DataPoint> trainBalanced = Preprocessor.oversample(trainNorm, NUM_CLASSES);
                runModelsAndAddPredictions(
                        deepCopy(trainBalanced), testNorm,
                        numFeatures, NUM_CLASSES,
                        dtAct3, lrAct3, knnAct3, 10, 5);

                // ACT 4: Balanced, Normalized & Tuned
                runModelsAndAddPredictions(
                        deepCopy(trainBalanced), testNorm,
                        numFeatures, NUM_CLASSES,
                        dtAct4, lrAct4, knnAct4, OPT_MAX_DEPTH, OPT_K);
            }
        }

        System.out.println("\n\nAll " + NUM_REPEATS + " repeats complete! Generating final reports...\n");

        System.out.println("\n\n<<<<<<<< ACT 1: IMBALANCED DATA (AVERAGE OF " + K_FOLDS + " FOLDS) >>>>>>>>");
        dtAct1.printReport("Decision Tree");
        lrAct1.printReport("Logistic Regression");
        knnAct1.printReport("KNN (k=5)");

        System.out.println("\n\n<<<<<<<< ACT 2: NORMALIZED DATA (AVERAGE OF " + K_FOLDS + " FOLDS) >>>>>>>>");
        dtAct2.printReport("Decision Tree");
        lrAct2.printReport("Logistic Regression");
        knnAct2.printReport("KNN (k=5)");

        System.out.println("\n\n<<<<<<<< ACT 3: BALANCED & NORMALIZED (Default Params) >>>>>>>>");
        dtAct3.printReport("Decision Tree (Depth=10)");
        lrAct3.printReport("Logistic Regression");
        knnAct3.printReport("KNN (k=5)");

        System.out.println("\n\n<<<<<<<< ACT 4: FINAL OPTIMIZED MODELS (Tuned Params) >>>>>>>>");
        dtAct4.printReport("OPTIMIZED Decision Tree (Depth=5)");
        lrAct4.printReport("OPTIMIZED Logistic Regression");
        knnAct4.printReport("OPTIMIZED KNN (K=21)");
    }

    // =========================================================================
    // OPTION 3: PREDICT FOR A NEW STUDENT
    // =========================================================================
    private static void runPrediction(List<DataPoint> allData, int numFeatures, Scanner scanner) {
        System.out.println("\n\n==========================================================");
        System.out.println("        OPTION 3: PREDICT STRESS FOR A NEW STUDENT       ");
        System.out.println("==========================================================");
        System.out.println("Please enter the student's details below.");
        System.out.println("(Models: Act 4 pipeline - Balanced, Normalized, Tuned params)");
        System.out.println("----------------------------------------------------------");

        // --- AGE ---
        System.out.println("\nAge Group:");
        System.out.println("  1 -> Below 18   (encoded as 17.0)");
        System.out.println("  2 -> 18-22      (encoded as 20.0)");
        System.out.println("  3 -> 23-26      (encoded as 24.5)");
        System.out.println("  4 -> 27-30      (encoded as 28.5)");
        System.out.println("  5 -> Above 30   (encoded as 31.0)");
        double age = readChoice(scanner, "Select Age Group (1-5): ",
                new double[]{17.0, 20.0, 24.5, 28.5, 31.0});

        // --- GENDER ---
        System.out.println("\nGender:");
        System.out.println("  1 -> Female            (encoded as 0.0)");
        System.out.println("  2 -> Male              (encoded as 1.0)");
        System.out.println("  3 -> Prefer not to say (encoded as 2.0)");
        double gender = readChoice(scanner, "Select Gender (1-3): ",
                new double[]{0.0, 1.0, 2.0});

        // --- ACADEMIC YEAR (display only) ---
        System.out.println("\nAcademic Year (for display only - not used in prediction):");
        System.out.println("  1 -> First Year");
        System.out.println("  2 -> Second Year");
        System.out.println("  3 -> Third Year");
        System.out.println("  4 -> Fourth Year");
        System.out.println("  5 -> Other");
        String[] acadNames = {"First Year", "Second Year", "Third Year", "Fourth Year", "Other"};
        int acadChoice = readIntChoice(scanner, "Select Academic Year (1-5): ", 1, 5);
        String acadYear = acadNames[acadChoice - 1];

        // --- CGPA ---
        System.out.println("\nCurrent CGPA:");
        System.out.println("  1 -> Below 2.50        (encoded as 2.490)");
        System.out.println("  2 -> 2.50 - 2.99       (encoded as 2.745)");
        System.out.println("  3 -> 3.00 - 3.49       (encoded as 3.245)");
        System.out.println("  4 -> 3.50 - 3.99       (encoded as 3.745)");
        System.out.println("  5 -> 4.00              (encoded as 4.000)");
        double cgpa = readChoice(scanner, "Select CGPA range (1-5): ",
                new double[]{2.49, 2.745, 3.245, 3.745, 4.0});

        // --- WAIVER / SCHOLARSHIP (display only) ---
        System.out.println("\nWaiver / Scholarship (for display only - not used in prediction):");
        System.out.println("  1 -> No");
        System.out.println("  2 -> Yes");
        int waiverChoice = readIntChoice(scanner, "Select (1-2): ", 1, 2);
        String waiver = (waiverChoice == 1) ? "No" : "Yes";

        // --- ANXIETY SCORE ---
        System.out.println("\nAnxiety Score (GAD-7 scale: 0 to 21):");
        double anxiety = readDouble(scanner, "Enter Anxiety Score: ", 0.0, 21.0);

        // --- DEPRESSION SCORE ---
        System.out.println("\nDepression Score (PHQ-9 scale: 0 to 27):");
        double depression = readDouble(scanner, "Enter Depression Score: ", 0.0, 27.0);

        // --- FEATURE ARRAY (must match DataLoader order: age, gender, cgpa, anxiety, depression) ---
        double[] rawFeatures = new double[]{age, gender, cgpa, anxiety, depression};

        // --- PRINT INPUT SUMMARY ---
        System.out.println("\n----------------------------------------------------------");
        System.out.println("  INPUT SUMMARY");
        System.out.println("----------------------------------------------------------");
        System.out.printf("  %-25s : %s%n",   "Age Group",           getAgeLabel(age));
        System.out.printf("  %-25s : %s%n",   "Gender",              getGenderLabel(gender));
        System.out.printf("  %-25s : %s%n",   "Academic Year",       acadYear);
        System.out.printf("  %-25s : %.3f%n", "CGPA (encoded)",      cgpa);
        System.out.printf("  %-25s : %s%n",   "Waiver/Scholarship",  waiver);
        System.out.printf("  %-25s : %.1f%n", "Anxiety Score",       anxiety);
        System.out.printf("  %-25s : %.1f%n", "Depression Score",    depression);
        System.out.println("----------------------------------------------------------");

        // --- TRAIN ACT 4 MODELS ON FULL DATASET ---
        System.out.println("\nTraining Act 4 models on the full dataset, please wait...");

        // Normalize: fit on full training data, apply same scaling to the new input point
        List<DataPoint> trainData = deepCopy(allData);
        DataPoint newPoint = new DataPoint(rawFeatures.clone(), -1); // label -1 = unknown
        List<DataPoint> singlePointList = new ArrayList<>();
        singlePointList.add(newPoint);
        Preprocessor.normalize(trainData, singlePointList);
        DataPoint normalizedPoint = singlePointList.get(0);

        // Oversample the normalized training data
        List<DataPoint> balancedTrain = Preprocessor.oversample(trainData, NUM_CLASSES);

        // Train the 3 Act 4 models
        DecisionTree dt = new DecisionTree(OPT_MAX_DEPTH, 2);
        dt.train(balancedTrain);

        LogisticRegression lr = new LogisticRegression(numFeatures, NUM_CLASSES, 0.01, 500);
        lr.train(balancedTrain);

        KNN knn = new KNN(OPT_K);
        knn.train(balancedTrain);

        // --- INDIVIDUAL PREDICTIONS ---
        int dtPred  = dt.predict(normalizedPoint);
        int lrPred  = lr.predict(normalizedPoint);
        int knnPred = knn.predict(normalizedPoint);

        // --- MAJORITY VOTE ---
        int[] votes = new int[NUM_CLASSES];
        votes[dtPred]++;
        votes[lrPred]++;
        votes[knnPred]++;

        int finalPrediction = 0;
        int maxVotes = -1;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                finalPrediction = i;
            }
        }

        // --- PRINT RESULTS ---
        System.out.println("\n==========================================================");
        System.out.println("                  PREDICTION RESULTS                     ");
        System.out.println("==========================================================");
        System.out.printf("  %-35s : %s%n", "Decision Tree predicts",       STRESS_LABELS[dtPred]);
        System.out.printf("  %-35s : %s%n", "Logistic Regression predicts", STRESS_LABELS[lrPred]);
        System.out.printf("  %-35s : %s%n", "KNN (k=21) predicts",          STRESS_LABELS[knnPred]);
        System.out.println("----------------------------------------------------------");
        System.out.printf("  %-35s : %s  (%d/3 votes)%n",
                ">>> FINAL PREDICTION (Majority Vote)",
                STRESS_LABELS[finalPrediction],
                maxVotes);
        System.out.println("==========================================================");
    }

    // =========================================================================
    // INPUT HELPER METHODS
    // =========================================================================

    /* Reads a 1-based menu choice and returns the matching value from the array. */
    private static double readChoice(Scanner scanner, String prompt, double[] values) {
        while (true) {
            System.out.print(prompt);
            try {
                int choice = Integer.parseInt(scanner.nextLine().trim());
                if (choice >= 1 && choice <= values.length) {
                    return values[choice - 1];
                }
            } catch (NumberFormatException ignored) {}
            System.out.println("Invalid input. Please enter a number between 1 and " + values.length + ".");
        }
    }

    /* Reads an integer choice within [min, max] inclusive. */
    private static int readIntChoice(Scanner scanner, String prompt, int min, int max) {
        while (true) {
            System.out.print(prompt);
            try {
                int val = Integer.parseInt(scanner.nextLine().trim());
                if (val >= min && val <= max) return val;
            } catch (NumberFormatException ignored) {}
            System.out.println("Invalid input. Please enter a number between " + min + " and " + max + ".");
        }
    }

    /* Reads a double value within [min, max] inclusive. */
    private static double readDouble(Scanner scanner, String prompt, double min, double max) {
        while (true) {
            System.out.print(prompt);
            try {
                double val = Double.parseDouble(scanner.nextLine().trim());
                if (val >= min && val <= max) return val;
            } catch (NumberFormatException ignored) {}
            System.out.println("Invalid input. Please enter a number between " + (int)min + " and " + (int)max + ".");
        }
    }

    /* Returns a readable age label from the encoded midpoint. */
    private static String getAgeLabel(double encodedAge) {
        if (encodedAge == 17.0) return "Below 18";
        if (encodedAge == 20.0) return "18-22";
        if (encodedAge == 24.5) return "23-26";
        if (encodedAge == 28.5) return "27-30";
        return "Above 30";
    }

    /** Returns a readable gender label from the encoded value. */
    private static String getGenderLabel(double encodedGender) {
        if (encodedGender == 0.0) return "Female";
        if (encodedGender == 1.0) return "Male";
        return "Prefer not to say";
    }

    // =========================================================================
    // SHARED HELPER METHODS (unchanged from original)
    // =========================================================================

    /**
     * Trains all 3 models and records their predictions into the provided confusion matrices.
     */
    private static void runModelsAndAddPredictions(
            List<DataPoint> trainSet, List<DataPoint> testSet,
            int numFeat, int numClass,
            ConfusionMatrix dtMat, ConfusionMatrix lrMat, ConfusionMatrix knnMat,
            int maxDepth, int k) {

        DecisionTree tree = new DecisionTree(maxDepth, 2);
        tree.train(trainSet);

        LogisticRegression lr = new LogisticRegression(numFeat, numClass, 0.01, 500);
        lr.train(trainSet);

        KNN knn = new KNN(k);
        knn.train(trainSet);

        for (DataPoint tp : testSet) {
            int actual = tp.getLabel();
            dtMat.addPrediction(actual,  tree.predict(tp));
            lrMat.addPrediction(actual,  lr.predict(tp));
            knnMat.addPrediction(actual, knn.predict(tp));
        }
    }

    /*
     * Creates a deep copy of a list of DataPoints.
     */
    private static List<DataPoint> deepCopy(List<DataPoint> original) {
        List<DataPoint> copy = new ArrayList<>();
        for (DataPoint dp : original) {
            // Preserve metadata fields so group tables remain accurate after copies
            copy.add(new DataPoint(dp.getFeatures().clone(), dp.getLabel(),
                    dp.getAcademicYear(), dp.getWaiver()));
        }
        return copy;
    }
}