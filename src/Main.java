import data.DataLoader;
import data.DataPoint;
import data.Preprocessor;
import evaluation.ConfusionMatrix;
import models.decisionTree.DecisionTree;
import models.knn.KNN;
import models.logisticRegression.LogisticRegression;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        System.out.println("--- Mental Health Stress Prediction: 1-Run Averaged Report ---");

        // --- GLOBAL SETUP ---
        DataLoader loader = new DataLoader();
        String filePath = "Processed.csv"; // Make sure this matches your file name
        List<DataPoint> masterData = loader.loadData(filePath);
        int numFeatures = masterData.get(0).getFeatureCount();
        int numClasses = 3;

        int NUM_RUNS = 1; // We will run the experiment 250 times to get stable averages

        // --- MASTER MATRICES (To accumulate results over all runs) ---
        // Act 1
        ConfusionMatrix dtAct1 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct1 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct1 = new ConfusionMatrix(numClasses);
        // Act 2
        ConfusionMatrix dtAct2 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct2 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct2 = new ConfusionMatrix(numClasses);
        // Act 3
        ConfusionMatrix dtAct3 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct3 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct3 = new ConfusionMatrix(numClasses);

        // NEW: Act 5 (Optimized Models)
        ConfusionMatrix dtAct5 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct5 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct5 = new ConfusionMatrix(numClasses);

        // Act 4: Tuning Arrays
        int[] kValues = {1, 3, 5, 7, 9, 11, 15, 21, 25};
        ConfusionMatrix[] knnTuningMatrices = new ConfusionMatrix[kValues.length];
        for(int i=0; i<kValues.length; i++) knnTuningMatrices[i] = new ConfusionMatrix(numClasses);

        int[] depthValues = {2, 3, 5, 7, 10, 12, 15};
        ConfusionMatrix[] dtTuningMatrices = new ConfusionMatrix[depthValues.length];
        for(int i=0; i<depthValues.length; i++) dtTuningMatrices[i] = new ConfusionMatrix(numClasses);


        // =================================================================================
        // THE MEGA-LOOP: Run the entire pipeline 250 times
        // =================================================================================
        System.out.println("Running " + NUM_RUNS + " randomized experiments. Please wait...");

        for (int run = 1; run <= NUM_RUNS; run++) {
            System.out.print("Running iteration " + run + " / " + NUM_RUNS + "...\r");

            // 0. Create a fresh, deep copy of the data for this run
            List<DataPoint> currentRunData = deepCopy(masterData);

            // 1. Split Data
            List<List<DataPoint>> splitData = Preprocessor.splitData(currentRunData, 0.8);
            List<DataPoint> trainRaw = splitData.get(0);
            List<DataPoint> testData = splitData.get(1); // Test set remains locked for this run

            // --- ACT 1: IMBALANCED & UN-NORMALIZED ---
            runModelsAndAddPredictions(trainRaw, testData, numFeatures, numClasses, dtAct1, lrAct1, knnAct1);

            // --- ACT 2: IMBALANCED & NORMALIZED ---
            List<DataPoint> trainNorm = deepCopy(trainRaw);
            List<DataPoint> testNorm = deepCopy(testData);
            Preprocessor.normalize(trainNorm, testNorm);
            runModelsAndAddPredictions(trainNorm, testNorm, numFeatures, numClasses, dtAct2, lrAct2, knnAct2);

            // --- ACT 3: BALANCED & NORMALIZED (Default Hyperparameters) ---
            List<DataPoint> trainBalanced = Preprocessor.oversample(trainNorm, numClasses);
            runModelsAndAddPredictions(trainBalanced, testNorm, numFeatures, numClasses, dtAct3, lrAct3, knnAct3);

            // --- NEW! ACT 5: BALANCED, NORMALIZED & TUNED (Optimal Hyperparameters) ---
            // We use the optimal settings found from your previous tuning reports: Depth=5, K=21
            DecisionTree tunedTree = new DecisionTree(5, 2);
            tunedTree.train(trainBalanced);

            LogisticRegression tunedLr = new LogisticRegression(numFeatures, numClasses, 0.01, 100);
            tunedLr.train(trainBalanced);

            KNN tunedKnn = new KNN(21);
            tunedKnn.train(trainBalanced);

            for (DataPoint tp : testNorm) {
                int actual = tp.getLabel();
                dtAct5.addPrediction(actual, tunedTree.predict(tp));
                lrAct5.addPrediction(actual, tunedLr.predict(tp));
                knnAct5.addPrediction(actual, tunedKnn.predict(tp));
            }

            // --- ACT 4: TUNING (On Balanced/Normalized data) ---
            // Tune KNN
            for (int i = 0; i < kValues.length; i++) {
                KNN tuningKnn = new KNN(kValues[i]);
                tuningKnn.train(trainBalanced);
                for (DataPoint tp : testNorm) {
                    knnTuningMatrices[i].addPrediction(tp.getLabel(), tuningKnn.predict(tp));
                }
            }
            // Tune Decision Tree
            for (int i = 0; i < depthValues.length; i++) {
                DecisionTree tuningTree = new DecisionTree(depthValues[i], 2);
                tuningTree.train(trainBalanced);
                for (DataPoint tp : testNorm) {
                    dtTuningMatrices[i].addPrediction(tp.getLabel(), tuningTree.predict(tp));
                }
            }
        }

        System.out.println("\n\nAll " + NUM_RUNS + " iterations complete! Generating Averaged Final Reports...\n");

        // =================================================================================
        // PRINT FINAL AVERAGED REPORTS
        // =================================================================================
        System.out.println("\n<<<<<<<< ACT 1: RAW IMBALANCED DATA (AVERAGE OF " + NUM_RUNS + " RUNS) >>>>>>>>");
        dtAct1.printReport("Decision Tree (Imbalanced)");
        lrAct1.printReport("Logistic Regression (Imbalanced)");
        knnAct1.printReport("KNN k=5 (Imbalanced)");

        System.out.println("\n\n<<<<<<<< ACT 2: NORMALIZED DATA (AVERAGE OF " + NUM_RUNS + " RUNS) >>>>>>>>");
        dtAct2.printReport("Decision Tree (Normalized)");
        lrAct2.printReport("Logistic Regression (Normalized)");
        knnAct2.printReport("KNN k=5 (Normalized)");

        System.out.println("\n\n<<<<<<<< ACT 3: BALANCED & NORMALIZED (Default Params) >>>>>>>>");
        dtAct3.printReport("Decision Tree (Balanced+Norm, Depth=10)");
        lrAct3.printReport("Logistic Regression (Balanced+Norm)");
        knnAct3.printReport("KNN (Balanced+Norm, K=5)");

        // NEW: Print Act 5 Results
        System.out.println("\n\n<<<<<<<< ACT 5: THE FINAL OPTIMIZED MODELS (Balanced, Norm, Tuned) >>>>>>>>");
        dtAct5.printReport("OPTIMIZED Decision Tree (Depth=5)");
        lrAct5.printReport("OPTIMIZED Logistic Regression");
        knnAct5.printReport("OPTIMIZED KNN (K=21)");

        System.out.println("\n\n<<<<<<<< ACT 4: TUNING TRENDS (AVERAGE OF " + NUM_RUNS + " RUNS) >>>>>>>>");

        System.out.println("\n--- KNN Tuning (K Values) ---");
        System.out.printf("%-5s | %-15s | %-15s | %-15s | %-15s\n", "K", "Overall Acc", "LOW Recall", "HIGH Recall", "MOD F1-Score");
        System.out.println("--------------------------------------------------------------------------");
        for (int i = 0; i < kValues.length; i++) {
            System.out.printf("%-5d | %-14.2f%% | %-14.2f%% | %-14.2f%% | %-14.2f%%\n",
                    kValues[i], knnTuningMatrices[i].getAccuracy()*100, knnTuningMatrices[i].getRecall(0)*100, knnTuningMatrices[i].getRecall(2)*100, knnTuningMatrices[i].getF1Score(1)*100);
        }

        System.out.println("\n--- Decision Tree Tuning (maxDepth) ---");
        System.out.printf("%-10s | %-15s | %-15s | %-15s | %-15s\n", "maxDepth", "Overall Acc", "LOW Recall", "HIGH Recall", "MOD F1-Score");
        System.out.println("-------------------------------------------------------------------------------");
        for (int i = 0; i < depthValues.length; i++) {
            System.out.printf("%-10d | %-14.2f%% | %-14.2f%% | %-14.2f%% | %-14.2f%%\n",
                    depthValues[i], dtTuningMatrices[i].getAccuracy()*100, dtTuningMatrices[i].getRecall(0)*100, dtTuningMatrices[i].getRecall(2)*100, dtTuningMatrices[i].getF1Score(1)*100);
        }
        System.out.println("\n\n<<<<<<<<<<<<<<< PHASE 5: FEATURE SELECTION ANALYSIS >>>>>>>>>>>>>>>");
        System.out.println("Proving why we use these features (Correlation with Stress Label):");
        System.out.println("---------------------------------------------------------------");

        String[] featureNames = {"Age", "Gender", "CGPA", "Anxiety", "Depression"};

        for (int i = 0; i < featureNames.length; i++) {
            double r = Preprocessor.calculateCorrelation(masterData, i);

            // Determine strength based on the "Distance from Zero" rule we learned
            String strength = (Math.abs(r) > 0.5) ? "STRONG" : (Math.abs(r) > 0.2 ? "MODERATE" : "WEAK");

            System.out.printf("Feature: %-12s | r = %-6.2f | Relationship: %s\n",
                    featureNames[i], r, strength);
        }
        System.out.println("===============================================================");
    }

    /**
     * Helper method to train all 3 models using DEFAULT params and record their predictions.
     */
    private static void runModelsAndAddPredictions(List<DataPoint> trainSet, List<DataPoint> testSet, int numFeat, int numClass, ConfusionMatrix dtMat, ConfusionMatrix lrMat, ConfusionMatrix knnMat) {
        DecisionTree tree = new DecisionTree(10, 2);
        tree.train(trainSet);
        LogisticRegression lr = new LogisticRegression(numFeat, numClass, 0.01, 100);
        lr.train(trainSet);
        KNN knn = new KNN(5); // Using your KNN constructor signature
        knn.train(trainSet);

        for (DataPoint tp : testSet) {
            int actual = tp.getLabel();
            dtMat.addPrediction(actual, tree.predict(tp));
            lrMat.addPrediction(actual, lr.predict(tp));
            knnMat.addPrediction(actual, knn.predict(tp));
        }
    }

    /**
     * Helper method to create a deep copy of the data.
     */
    private static List<DataPoint> deepCopy(List<DataPoint> original) {
        List<DataPoint> copy = new ArrayList<>();
        for (DataPoint dp : original) {
            copy.add(new DataPoint(dp.getFeatures().clone(), dp.getLabel()));
        }
        return copy;
    }
}