package evaluation;

public class ConfusionMatrix {

    // The matrix grid: matrix[actual_label][predicted_label]
    private final int[][] matrix;
    private final int numClasses;
    private int totalSamples;

    public ConfusionMatrix(int numClasses) {
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
        this.totalSamples = 0;
    }

    /**
     * Adds the result of a single prediction to the matrix.
     * This is the main way we populate the grid.
     */
    public void addPrediction(int actual, int predicted) {
        if (actual >= 0 && actual < numClasses && predicted >= 0 && predicted < numClasses) {
            matrix[actual][predicted]++;
            totalSamples++;
        }
    }

    /**
     * Calculates the Overall Accuracy.
     * Formula: (Sum of Diagonal) / (Total Samples)
     */
    public double getAccuracy() {
        if (totalSamples == 0) return 0.0;
        double correct = 0;
        for (int i = 0; i < numClasses; i++) {
            correct += matrix[i][i]; // Summing the main diagonal
        }
        return correct / totalSamples;
    }

    /**
     * Calculates the Precision for a single class.
     * "Of all the times we PREDICTED this class, how often were we right?"
     * Formula: TruePositives / (All Predictions for this Class)
     */
    public double getPrecision(int classIndex) {
        int truePositives = matrix[classIndex][classIndex];
        int allPredictedAsClass = 0;
        // To find all predictions for this class, we must sum DOWN the COLUMN.
        for (int i = 0; i < numClasses; i++) {
            allPredictedAsClass += matrix[i][classIndex];
        }
        if (allPredictedAsClass == 0) return 0.0;
        return (double) truePositives / allPredictedAsClass;
    }

    /**
     * Calculates the Recall for a single class.
     * "Of all the ACTUAL instances of this class, how many did we find?"
     * Formula: TruePositives / (All Actual Instances of this Class)
     */
    public double getRecall(int classIndex) {
        int truePositives = matrix[classIndex][classIndex];
        int allActualOfClass = 0;
        // To find all actual instances, we must sum ACROSS the ROW.
        for (int j = 0; j < numClasses; j++) {
            allActualOfClass += matrix[classIndex][j];
        }
        if (allActualOfClass == 0) return 0.0;
        return (double) truePositives / allActualOfClass;
    }

    /**
     * Calculates the F1-Score for a single class.
     * This is the harmonic mean of Precision and Recall.
     */
    public double getF1Score(int classIndex) {
        double precision = getPrecision(classIndex);
        double recall = getRecall(classIndex);
        if (precision + recall == 0) return 0.0;
        return 2 * (precision * recall) / (precision + recall);
    }

    /**
     * Prints a full, formatted report including the confusion matrix, overall accuracy,
     * and class-wise Precision, Recall, and F1-Score.
     * @param modelName The name of the model being evaluated (e.g., "Decision Tree").
     */
    public void printReport(String modelName) {
        String[] labels = {"LOW", "MODERATE", "HIGH"};

        System.out.println("\n==========================================================");
        System.out.println("          EVALUATION REPORT: " + modelName.toUpperCase());
        System.out.println("==========================================================");

        // --- Print the Confusion Matrix with better formatting ---
        System.out.println("\nConfusion Matrix (Rows=Actual, Cols=Predicted):");
        System.out.println("----------------------------------------------------------");
        System.out.printf("%-15s | %-10s | %-10s | %-10s\n", "ACTUAL", "PRED LOW", "PRED MOD", "PRED HIGH");
        System.out.println("----------------------------------------------------------");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%-15s | ", labels[i]);
            for (int j = 0; j < numClasses; j++) {
                System.out.printf("%-10d | ", matrix[i][j]);
            }
            System.out.println();
        }
        System.out.println("----------------------------------------------------------");

        // --- Print Overall Accuracy ---
        System.out.printf("\nOVERALL ACCURACY: %.2f%%\n", getAccuracy() * 100);

        // --- Print Class-wise Metrics ---
        System.out.println("\nClass-wise Metrics (to analyze bias):");
        System.out.println("----------------------------------------------------------");
        System.out.printf("%-15s | %-10s | %-10s | %-10s\n", "CLASS", "PRECISION", "RECALL", "F1-SCORE");
        System.out.println("----------------------------------------------------------");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%-15s | %-10.2f%% | %-10.2f%% | %-10.2f%%\n",
                    labels[i],
                    getPrecision(i) * 100,
                    getRecall(i) * 100,
                    getF1Score(i) * 100);
        }
        System.out.println("==========================================================");
    }
}