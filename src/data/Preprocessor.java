package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Preprocessor {

    /**
     * Splits the full dataset into a training set and a testing set.
     * @param data The full list of DataPoints.
     * @param trainSplitRatio The proportion of data for the training set (e.g., 0.8 for 80%).
     * @return A List containing two lists: the training set at index 0, and the testing set at index 1.
     */
    public static List<List<DataPoint>> splitData(List<DataPoint> data, double trainSplitRatio) {
        // Creating a mutable copy of the data to avoid modifying the original list.
        List<DataPoint> shuffledData = new ArrayList<>(data);

        // Randomly shuffle the data. This is crucial to ensure that the train and test sets
        // are representative of the overall data and not biased by any original ordering.
        Collections.shuffle(shuffledData);

        // Calculate the index where we will split the data.
        int splitIndex = (int) (shuffledData.size() * trainSplitRatio);

        // Create the training set as a sublist from the beginning to the split index.
        List<DataPoint> trainingSet = new ArrayList<>(shuffledData.subList(0, splitIndex));

        // Create the testing set as a sublist from the split index to the end.
        List<DataPoint> testingSet = new ArrayList<>(shuffledData.subList(splitIndex, shuffledData.size()));

        // Return both sets in a container list.
        List<List<DataPoint>> result = new ArrayList<>();
        result.add(trainingSet);
        result.add(testingSet);
        return result;
    }
    //----------Oversampling-----------------------------
    /**
     * Balances the training data by oversampling the minority classes.
     * It makes random copies of minority class data points until all classes
     * have the same number of samples as the majority class.
     *
     * @param trainingData The imbalanced training set.
     * @param numClasses The total number of classes (e.g., 3).
     * @return A new, balanced list of DataPoints ready for training.
     */
    public static List<DataPoint> oversample(List<DataPoint> trainingData, int numClasses) {

        // 1. Create "buckets" for each class
        List<List<DataPoint>> classBuckets = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            classBuckets.add(new ArrayList<>());
        }

        // 2. Sort all the training data into their respective buckets
        for (DataPoint dp : trainingData) {
            int label = dp.getLabel();
            classBuckets.get(label).add(dp);
        }

        // 3. Find the size of the largest bucket (the majority class)
        int maxSize = 0;
        for (List<DataPoint> bucket : classBuckets) {
            if (bucket.size() > maxSize) {
                maxSize = bucket.size();
            }
        }

        // 4. Create a new list to hold our final balanced data
        List<DataPoint> balancedData = new ArrayList<>();
        Random rand = new Random();

        // 5. Fill the balanced list
        for (List<DataPoint> bucket : classBuckets) {
            if (bucket.isEmpty()) continue; // Safety check

            // First, add all the original students from this bucket
            balancedData.addAll(bucket);

            // Calculate how many photocopies we need to make to reach maxSize
            int numCopiesNeeded = maxSize - bucket.size();

            // Randomly select students from this bucket and add copies
            for (int i = 0; i < numCopiesNeeded; i++) {
                int randomIndex = rand.nextInt(bucket.size());
                balancedData.add(bucket.get(randomIndex));
            }
        }

        // 6. Shuffle the final deck so the model doesn't just see a block of duplicates at the end
        Collections.shuffle(balancedData);

        return balancedData;
    }
    /**
     * Normalizes the features of a dataset using Min-Max scaling.
     * It scales all feature values to be between 0.0 and 1.0.
     *
     * @param trainingSet The training data, used to find the min/max for each feature.
     * @param testingSet The testing data, which will be scaled using the min/max from the training set.
     */
    public static void normalize(List<DataPoint> trainingSet, List<DataPoint> testingSet) {
        if (trainingSet.isEmpty()) {
            return; // Cannot normalize without training data
        }

        int numFeatures = trainingSet.get(0).getFeatureCount();
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];

        // Initialize min and max arrays with values from the first data point
        for (int i = 0; i < numFeatures; i++) {
            minValues[i] = trainingSet.get(0).getFeature(i);
            maxValues[i] = trainingSet.get(0).getFeature(i);
        }

        // --- Step 1: Find the min and max for each feature from the TRAINING SET ONLY ---
        for (DataPoint dp : trainingSet) {
            for (int i = 0; i < numFeatures; i++) {
                if (dp.getFeature(i) < minValues[i]) {
                    minValues[i] = dp.getFeature(i);
                }
                if (dp.getFeature(i) > maxValues[i]) {
                    maxValues[i] = dp.getFeature(i);
                }
            }
        }

        // --- Step 2: Apply the normalization to the TRAINING SET ---
        for (DataPoint dp : trainingSet) {
            for (int i = 0; i < numFeatures; i++) {
                double range = maxValues[i] - minValues[i];
                if (range != 0) {
                    // Get the original value
                    double originalValue = dp.getFeature(i);
                    // Calculate and set the new normalized value
                    dp.getFeatures()[i] = (originalValue - minValues[i]) / range;
                } else {
                    // If range is 0, all values are the same, so normalize to 0 or 0.5
                    dp.getFeatures()[i] = 0.0;
                }
            }
        }

        // --- Step 3: Apply the EXACT SAME normalization to the TESTING SET ---
        // We use the min/max values learned from the training data to prevent data leakage.
        for (DataPoint dp : testingSet) {
            for (int i = 0; i < numFeatures; i++) {
                double range = maxValues[i] - minValues[i];
                if (range != 0) {
                    dp.getFeatures()[i] = (dp.getFeature(i) - minValues[i]) / range;
                } else {
                    dp.getFeatures()[i] = 0.0;
                }
            }
        }
    }
    /**
     * Calculates the Pearson Correlation Coefficient between a feature and the target label.
     * Formula: r = SXY / sqrt(SSX * SSY)
     */
    public static double calculateCorrelation(List<DataPoint> data, int featureIndex) {
        int n = data.size();
        if (n == 0) return 0.0;

        double sumX = 0, sumY = 0, sumXY = 0;
        double sumX2 = 0, sumY2 = 0;

        for (DataPoint dp : data) {
            double x = dp.getFeature(featureIndex);
            double y = (double) dp.getLabel();

            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
            sumY2 += y * y;
        }

        double numerator = (n * sumXY) - (sumX * sumY);
        double denominator = Math.sqrt(((n * sumX2) - (sumX * sumX)) * ((n * sumY2) - (sumY * sumY)));

        if (denominator == 0) return 0.0;
        return numerator / denominator;
    }
}
