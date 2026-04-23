package models.logisticRegression;

import data.DataPoint;
import java.util.List;
import java.util.Random;

/*
 * A from-scratch implementation of Multi-class Logistic Regression (Softmax Regression).
 */
public class LogisticRegression {

    private double[][] weights;
    private double[] biases;
    private final double learningRate;
    private final int epochs;

    // the constructor
    public LogisticRegression(int numFeatures, int numClasses, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.weights = new double[numFeatures][numClasses];
        this.biases = new double[numClasses];

        Random rand = new Random();
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numClasses; j++) {
                this.weights[i][j] = (rand.nextDouble() - 0.5) / 50.0; // (rand.nextDouble() - 0.5) scales to -.5 to +.5 and dividing to 50 gets a more small value
            }
        }
    }

    // the main training method of Stochastic Gradient Descent
    public void train(List<DataPoint> trainingData) {
        int numFeatures = weights.length;
        int numClasses = biases.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (DataPoint dp : trainingData) {
                double[] features = dp.getFeatures();

                // step 1: forward Pass
                // calculating Z and then P using softmax func
                double[] scores = calculateScores(features);
                double[] probabilities = softmax(scores);

                // step 2: calculating Error(E)
                double[] errorSignal = new double[numClasses];
                int trueLabel = dp.getLabel();
                for (int j = 0; j < numClasses; j++) {
                    double y_true = (j == trueLabel) ? 1.0 : 0.0;
                    errorSignal[j] = probabilities[j] - y_true;
                }

                // step 3 & 4: updating Parameters
                //b_new = b_old - learning_rate * grad_b -> grad_b = E
                for (int j = 0; j < numClasses; j++) {
                    biases[j] -= learningRate * errorSignal[j];
                }
                //W_new = W_old - learning_rate * grad_W -> grad_W = X_transposed • E
                for (int i = 0; i < numFeatures; i++) {
                    for (int j = 0; j < numClasses; j++) {
                        weights[i][j] -= learningRate * features[i] * errorSignal[j];
                    }
                }
            }
        }
    }

    /**
     * The "Forward Pass" for a single data point to make a prediction.
     * @param dataPoint The data point to classify.
     * @return The predicted class index (0, 1, or 2).
     */
    public int predict(DataPoint dataPoint) {
        // calculate the raw scores (Z) for each class ---
        double[] scores = calculateScores(dataPoint.getFeatures());

        // convert scores to probabilities (P) using softmax ---
        double[] probabilities = softmax(scores);

        // -find the class with the highest probability ---
        int bestClass = -1;
        double maxProbability = -1.0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProbability) {
                maxProbability = probabilities[i];
                bestClass = i;
            }
        }
        return bestClass;
    }

    // =================================================================
    // HELPER METHODS
    // =================================================================

    // calculate the raw scores. (Z = X • W + b)
    private double[] calculateScores(double[] features) {
        double[] scores = new double[biases.length];
        for (int j = 0; j < biases.length; j++) {
            scores[j] = biases[j];
            for (int i = 0; i < features.length; i++) {
                scores[j] += features[i] * weights[i][j];
            }
        }
        return scores;
    }

    // to calculate the softmax function values for prabability
    private double[] softmax(double[] scores) {
        double[] probabilities = new double[scores.length];
        double maxScore = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            probabilities[i] = Math.exp(scores[i] - maxScore);
            sum += probabilities[i];
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
    }
}