package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Preprocessor {

    // the full list of data points , splits the data into training and testing set
    // index 0 -> training set, index 1 -> testing set
    public static List<List<DataPoint>> splitData(List<DataPoint> data, double trainSplitRatio) {
        // a mutable copy of the data to avoid modifying the original list.
        List<DataPoint> shuffledData = new ArrayList<>(data);

        // shuffling the full dataset
        Collections.shuffle(shuffledData);

        // calculating the index where we will split the data.
        int splitIndex = (int) (shuffledData.size() * trainSplitRatio);

        // create the training set as a sublist from the beginning to the split index.
        List<DataPoint> trainingSet = new ArrayList<>(shuffledData.subList(0, splitIndex));

        // create the testing set as a sublist from the split index to the end.
        List<DataPoint> testingSet = new ArrayList<>(shuffledData.subList(splitIndex, shuffledData.size()));

        // returning both sets in a container list.
        List<List<DataPoint>> result = new ArrayList<>();
        result.add(trainingSet);
        result.add(testingSet);
        return result;
    }
    //----------Oversampling-----------------------------
    // balances the traning data by oversampling the minority classes
    // finds out the majority class and for minority classes , randomly picks data 1 by 1 to have same number of data as majority class
    public static List<DataPoint> oversample(List<DataPoint> trainingData, int numClasses) {

        // creating "buckets" for each class
        List<List<DataPoint>> classBuckets = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            classBuckets.add(new ArrayList<>());
        }

        // sort all the training data into their respective buckets linearly
        for (DataPoint dp : trainingData) {
            int label = dp.getLabel();
            classBuckets.get(label).add(dp);
        }

        // find the size of the largest bucket (the majority class) -> moderate class here
        int maxSize = 0;
        for (List<DataPoint> bucket : classBuckets) {
            if (bucket.size() > maxSize) {
                maxSize = bucket.size();
            }
        }

        // creating a new list to hold our final balanced data
        List<DataPoint> balancedData = new ArrayList<>();
        Random rand = new Random();

        // fill the balanced list
        for (List<DataPoint> bucket : classBuckets) {
            if (bucket.isEmpty()) continue;

            // at first adding the original students from each bucket
            balancedData.addAll(bucket);

            // calculating how many photocopies needed to make to reach maxSize
            int numCopiesNeeded = maxSize - bucket.size();

            // randomly selecting students from this bucket and add copies
            for (int i = 0; i < numCopiesNeeded; i++) {
                int randomIndex = rand.nextInt(bucket.size());
                balancedData.add(bucket.get(randomIndex));
            }
        }

        // shuffling the final deck so the model doesn't just see a block of duplicates at the end
        Collections.shuffle(balancedData);

        return balancedData;
    }

    // normalizes the dataset using min-max scaling
    // finding min/max for training dataset and applying this for both dataset
    public static void normalize(List<DataPoint> trainingSet, List<DataPoint> testingSet) {
        if (trainingSet.isEmpty()) {
            return;
        }

        int numFeatures = trainingSet.get(0).getFeatureCount();
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];

        // initialize min and max arrays with values from the first data point
        for (int i = 0; i < numFeatures; i++) {
            minValues[i] = trainingSet.get(0).getFeature(i);
            maxValues[i] = trainingSet.get(0).getFeature(i);
        }

        // finding the min and max for each feature from the ------- TRAINING SET ONLY ---
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

        // apply the normalization to the TRAINING SET ---
        for (DataPoint dp : trainingSet) {
            for (int i = 0; i < numFeatures; i++) {
                double range = maxValues[i] - minValues[i];
                if (range != 0) {
                    // get the original value
                    double originalValue = dp.getFeature(i);
                    // calculate and set the new normalized value
                    dp.getFeatures()[i] = (originalValue - minValues[i]) / range;
                } else {
                    // If range is 0, all values are the same, so normalize to 0
                    dp.getFeatures()[i] = 0.0;
                }
            }
        }

        // applyng the EXACT SAME normalization to the TESTING SET ---
        // using the min/max values learned from the training data to prevent data leakage
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

    // calculating Pearson Correlation Coefficient between a feature and the target label
    // Formula: r = SXY / sqrt(SSX * SSY)
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

    // List containing two lists: the training set at index 0, and the testing set at index 1.
    public static List<List<DataPoint>> getKFoldSplit(List<DataPoint> allData, int numFolds, int foldIndex) {
        int totalSize = allData.size();
        int foldSize = totalSize / numFolds;

        // identifying the start and end indices of the test fold ---
        int testStartIndex = foldIndex * foldSize;
        // the end index should not go past the end of the list.
        int testEndIndex = Math.min(testStartIndex + foldSize, totalSize);

        // creating the testing set
        // sublist for the current fold.
        List<DataPoint> testingSet = new ArrayList<>(allData.subList(testStartIndex, testEndIndex));

        // create the training set ---
        // it's everything EXCEPT the testing set.
        List<DataPoint> trainingSet = new ArrayList<>();
        // adding all data *before* the test set.
        trainingSet.addAll(allData.subList(0, testStartIndex));
        // adding all data *after* the test set.
        trainingSet.addAll(allData.subList(testEndIndex, totalSize));

        // returning both sets in a container list ---
        List<List<DataPoint>> result = new ArrayList<>();
        result.add(trainingSet);
        result.add(testingSet);
        return result;
    }

    // generates descriptive statistics (Mean and Standard Deviation) for a feature.
    public static String getStats(List<DataPoint> data, int featureIndex) {
        int n = data.size();
        if (n == 0) return "0.00 ± 0.00";

        double sum = 0;
        for (DataPoint dp : data) {
            sum += dp.getFeature(featureIndex);
        }
        double mean = sum / n;

        double squaredDiffSum = 0;
        for (DataPoint dp : data) {
            double diff = dp.getFeature(featureIndex) - mean;
            squaredDiffSum += diff * diff;
        }
        double sd = Math.sqrt(squaredDiffSum / n);

        return String.format("%.2f ± %.2f", mean, sd);
    }
}
