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

public class Main {

    public static void main(String[] args) {
        System.out.println("--- Mental Health Stress Prediction: 10-Fold Cross-Validated Report ---");

        // --- GLOBAL SETUP ---
        DataLoader loader = new DataLoader();
        String filePath = "MentalHealth.csv";
        List<DataPoint> allData = loader.loadData(filePath);
        Collections.shuffle(allData);

        int numFeatures = allData.get(0).getFeatureCount();
        int numClasses = 3;
        int K_FOLDS = 10;

        // --- MASTER MATRICES for each ACT (To accumulate results over all 10 folds) ---
        // Act 1: Imbalanced
        ConfusionMatrix dtAct1 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct1 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct1 = new ConfusionMatrix(numClasses);
        // Act 2: Normalized
        ConfusionMatrix dtAct2 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct2 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct2 = new ConfusionMatrix(numClasses);
        // Act 3: Balanced & Normalized (Default Params)
        ConfusionMatrix dtAct3 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct3 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct3 = new ConfusionMatrix(numClasses);
        // Act 4: Optimized (Balanced, Normalized, Tuned Params)
        ConfusionMatrix dtAct4 = new ConfusionMatrix(numClasses);
        ConfusionMatrix lrAct4 = new ConfusionMatrix(numClasses);
        ConfusionMatrix knnAct4 = new ConfusionMatrix(numClasses);


        // =================================================================================
        // THE K-FOLD CROSS-VALIDATION LOOP
        // =================================================================================
        System.out.println("Running " + K_FOLDS + "-Fold Cross-Validation across all 4 Acts. Please wait...");

        for (int k = 0; k < K_FOLDS; k++) {
            System.out.print("Processing Fold " + (k + 1) + "/" + K_FOLDS + "...\r");

            // --- Get the Train/Test split for the current fold ---
            List<List<DataPoint>> splitData = Preprocessor.getKFoldSplit(allData, K_FOLDS, k);
            List<DataPoint> rawTrain = splitData.get(0);
            List<DataPoint> rawTest = splitData.get(1);

            // --- ACT 1: IMBALANCED & UN-NORMALIZED ---
            runModelsAndAddPredictions(deepCopy(rawTrain), deepCopy(rawTest), numFeatures, numClasses, dtAct1, lrAct1, knnAct1, 10, 5);

            // --- ACT 2: IMBALANCED & NORMALIZED ---
            List<DataPoint> trainNorm = deepCopy(rawTrain);
            List<DataPoint> testNorm = deepCopy(rawTest);
            Preprocessor.normalize(trainNorm, testNorm);
            runModelsAndAddPredictions(trainNorm, testNorm, numFeatures, numClasses, dtAct2, lrAct2, knnAct2, 10, 5);

            // --- ACT 3: BALANCED & NORMALIZED (Default Hyperparameters) ---
            List<DataPoint> trainBalanced = Preprocessor.oversample(trainNorm, numClasses);
            runModelsAndAddPredictions(trainBalanced, testNorm, numFeatures, numClasses, dtAct3, lrAct3, knnAct3, 10, 5);

            // --- ACT 4: BALANCED, NORMALIZED & TUNED (Optimal Hyperparameters) ---
            runModelsAndAddPredictions(trainBalanced, testNorm, numFeatures, numClasses, dtAct4, lrAct4, knnAct4, 5, 21); // Using depth=5, k=21

        } // End of K-Fold loop

        System.out.println("\n\nAll " + K_FOLDS + " folds complete! Generating final averaged reports...\n");

        // =================================================================================
        // PRINT FINAL AVERAGED REPORTS FOR EACH ACT
        // =================================================================================
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

    /**
     * Helper method to train all 3 models and record their predictions in the provided Master Matrices.
     * Now includes parameters for hyperparameter tuning.
     */
    private static void runModelsAndAddPredictions(List<DataPoint> trainSet, List<DataPoint> testSet, int numFeat, int numClass,
                                                   ConfusionMatrix dtMat, ConfusionMatrix lrMat, ConfusionMatrix knnMat,
                                                   int maxDepth, int k) {
        DecisionTree tree = new DecisionTree(maxDepth, 2);
        tree.train(trainSet);
        LogisticRegression lr = new LogisticRegression(numFeat, numClass, 0.01, 100);
        lr.train(trainSet);
        KNN knn = new KNN(k);
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