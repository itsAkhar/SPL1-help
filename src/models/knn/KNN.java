package models.knn;

import data.DataPoint;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class KNN {

    private final int k;
    private List<DataPoint> trainingData;

    // a helper class to store neighbor's distance and label
    private static class Neighbor {
        public double distance;
        public int label;

        public Neighbor(double distance, int label) {
            this.distance = distance;
            this.label = label;
        }
    }

    // the constructor
    public KNN(int k) {
        this.k = k;
        this.trainingData = null;
    }

    // trains only by storing data
    public void train(List<DataPoint> trainingData) {
        this.trainingData = trainingData;
    }

    // process:calculate distances to all training points -> sort -> take top K -> majority vote
    public int predict(DataPoint dataPoint) {
        if (trainingData == null) {
            throw new IllegalStateException("KNN model has not been trained yet. Call train() first.");
        }

        // calculate distance from the query point to every training point
        List<Neighbor> neighbors = new ArrayList<>();
        for (DataPoint trainPoint : this.trainingData) {
            double distance = euclideanDistance(dataPoint.getFeatures(), trainPoint.getFeatures());
            neighbors.add(new Neighbor(distance, trainPoint.getLabel()));
        }

        // sort all neighbors by distance
        neighbors.sort(Comparator.comparingDouble(n -> n.distance));

        // this prevents an IndexOutOfBoundsException in edge cases.
        int effectiveK = Math.min(this.k, neighbors.size());
        List<Neighbor> kNearest = neighbors.subList(0, effectiveK);

        // majority vote across the K nearest neighbors
        // index 0-> low , 1-> moderate , 2-> high
        int[] voteCounts = new int[3];
        for (Neighbor neighbor : kNearest) {
            int label = neighbor.label;
            if (label >= 0 && label < voteCounts.length) {
                voteCounts[label]++;
            }
        }

        // finding the majority vote
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

    //Euclidean Distance Formula: For two points, A and B, with n features: Distance = √[ (A₁-B₁)² + (A₂-B₂)² + ... + (Aₙ-Bₙ)² ]
    private double euclideanDistance(double[] featuresA, double[] featuresB) {
        double sumOfSquaredDifferences = 0.0;
        for (int i = 0; i < featuresA.length; i++) {
            double diff = featuresA[i] - featuresB[i];
            sumOfSquaredDifferences += diff * diff;
        }
        return Math.sqrt(sumOfSquaredDifferences);
    }
}