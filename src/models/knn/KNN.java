package models.knn;

import data.DataPoint;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class KNN {

    private final int k;
    private List<DataPoint> trainingData;

    // A private helper class to store a neighbor's info
    // This makes it easy to sort our neighbors by their distance.
    private static class Neighbor {
        public double distance;
        public int label;

        public Neighbor(double distance, int label) {
            this.distance = distance;
            this.label = label;
        }
    }

    public KNN(int k) {
        this.k = k;
        this.trainingData = null;
    }

    public void train(List<DataPoint> trainingData) {
        this.trainingData = trainingData;
    }

    public int predict(DataPoint dataPoint) {
        if (trainingData == null) {
            throw new IllegalStateException("KNN model has not been trained yet. Call train() first.");
        }

        List<Neighbor> neighbors = new ArrayList<>();
        for (DataPoint trainPoint : this.trainingData) {
            double distance = euclideanDistance(dataPoint.getFeatures(), trainPoint.getFeatures());
            neighbors.add(new Neighbor(distance, trainPoint.getLabel()));
        }

        // Sort the neighbors by distance and find the top K
        // I use a comparator to tell the sort function to order by the 'distance' field.
        neighbors.sort(Comparator.comparingDouble(n -> n.distance));

        List<Neighbor> kNearest = neighbors.subList(0, this.k);

        // Take a majority vote
        // Assuming 3 classes (0, 1, 2), create an array to store the vote counts.
        int[] voteCounts = new int[3];

        // Loop through the K nearest neighbors and increment the vote count for their label.
        for (Neighbor neighbor : kNearest) {
            int label = neighbor.label;
            if (label >= 0 && label < voteCounts.length) {
                voteCounts[label]++;
            }
        }

        int majorityLabel = -1;
        int maxVotes = -1;
        for (int i = 0; i < voteCounts.length; i++) {
            if (voteCounts[i] > maxVotes) {
                maxVotes = voteCounts[i];
                majorityLabel = i;
            }
        }


        return majorityLabel;
    }
    // HELPER METHOD FOR DISTANCE CALCULATION
    private double euclideanDistance(double[] featuresA, double[] featuresB) {
        double sumOfSquaredDifferences = 0.0;
        for (int i = 0; i < featuresA.length; i++) {
            double diff = featuresA[i] - featuresB[i];
            sumOfSquaredDifferences += diff * diff;
        }
        // The final distance is the square root of the sum.
        return Math.sqrt(sumOfSquaredDifferences);
    }
}