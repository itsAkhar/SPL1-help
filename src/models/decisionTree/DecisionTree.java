package models.decisionTree;

import data.DataPoint;

import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

// the main class that is responsible for building the tree (training) and using it to make predictions
public class DecisionTree {

    // 'root' is the very first node of the tree
    // It is null until the train() method is called
    private Node root;

    // a hyperparameter to control the maximum depth of the tree
    // crucial for preventing overfitting
    private final int maxDepth;

    // a hyperparameter that sets a minimum number of data points required to attempt a split
    // prevents the tree from branching for very small group of data
    private final int minSamplesSplit;

    // the constructor for the tree
    public DecisionTree(int maxDepth, int minSamplesSplit) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
    }

    // takes the training data and begins the recursive tree-building process
    public void train(List<DataPoint> trainingData) {
        // the entire, fully-assembled tree structure is returned by buildTree and stored in 'root' field.
        this.root = buildTree(trainingData, 0);
    }

    // the main recursive method that builds the tree
    private Node buildTree(List<DataPoint> data, int currentDepth) {
        /* 3 base cases:
           1.reached the maxDepth
           2.reached minSamplesSplit
           3.the group of data is pure or not
         */
        if (currentDepth >= maxDepth || data.size() < minSamplesSplit || isPure(data)) {
            // any of the base case meets -> making a leaf node with the most common prediction
            int leafPrediction = majorityVote(data);
            return new Node(leafPrediction);
        }

        // Recursive step : finding the best split and continue -> the method that works all important work is findBestSplit()
        BestSplitResult bestSplit = findBestSplit(data);
        // base case: 4. info gain <= 0
        if (bestSplit.getGain() <= 0) {
            int leafPrediction = majorityVote(data);
            return new Node(leafPrediction);
        }

        // recursively call this function to build the "yes" (left) branch
        Node leftChild = buildTree(bestSplit.getLeftData(), currentDepth + 1);

        // recursively call this function to build the "no" (right) branch
        Node rightChild = buildTree(bestSplit.getRightData(), currentDepth + 1);

        // returning the node with best question and left + right subtree
        return new Node(bestSplit.getCondition(), leftChild, rightChild);
    }

    // the main public method to make a prediction on a new, unseen data point
    // returns predicted label
    public int predict(DataPoint dataPoint) {
        // start the recursive walk down the tree, beginning at the root.
        return traverseTree(dataPoint.getFeatures(), this.root);
    }

    // the private recursive method that walks down the tree to find a prediction
    private int traverseTree(double[] features, Node node) {
        // the base case : if the current node is a leaf , return the prediction
        if (node.isLeaf()) {
            return node.getPrediction();
        }
        // recursive step: it is a decision node. using the nodes split condition to check students feature
        if (node.getSplitCondition().matches(features)) {
            // if the condition is true, continue traversing down the LEFT child.
            return traverseTree(features, node.getLeftChild());
        } else {
            // if the condition is false, continue traversing down the RIGHT child.
            return traverseTree(features, node.getRightChild());
        }
    }

    // helper methods

    // Formula: The formula is 1 - Σ (pᵢ)², where pᵢ is the proportion (percentage) of each class i in the group.
    private double calculateGini(List<DataPoint> data) {
        if (data.isEmpty()) {
            return 0.0;
        }
        // array to hold the counts of each class -> low, moderate, high
        int[] classCounts = new int[3];

        // counting the occurence of each label
        for (DataPoint dp : data) {
            int label = dp.getLabel();
            if (label >= 0 && label < classCounts.length) {
                classCounts[label]++;
            }
        }

        // calculating the sum of squared proportions.
        double sumOfSquares = 0.0;
        int totalSamples = data.size();
        for (int count : classCounts) {
            if (count > 0) {
                double proportion = (double) count / totalSamples;
                sumOfSquares += proportion * proportion;
            }
        }
        return 1.0 - sumOfSquares; // final gini formula
    }

    // finds the most frequent class label in a list of data
    // returns the label of majority class
    private int majorityVote(List<DataPoint> data) {
        if (data.isEmpty()) {
            return -1; // list empty
        }

        // array to store the counts of each label
        int[] classCounts = new int[3];
        for (DataPoint dp : data) {
            int label = dp.getLabel();
            if (label >= 0 && label < classCounts.length) {
                classCounts[label]++;
            }
        }

        // finding the index in the array that has the highest count.
        int majorityLabel = -1;
        int maxCount = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > maxCount) {
                maxCount = classCounts[i];
                majorityLabel = i;
            }
        }
        return majorityLabel;
    }

    /*
     * This is the core of the algorithm. It iterates through every possible question (every feature and every unique value)
     * to find the one that results in the highest Information Gain (the biggest reduction in Gini impurity).
     * data -> The current list of data points to be split.
     * returns -> A BestSplitResult object containing the best question and the resulting data subgroups.
     */
    private BestSplitResult findBestSplit(List<DataPoint> data) {
        double bestGain = 0.0;
        SplitCondition bestCondition = null;
        List<DataPoint> bestLeftData = null;
        List<DataPoint> bestRightData = null;

        // calculating the impurity of the current group of data before splitting
        double parentGini = calculateGini(data);
        if (data.isEmpty()) {
            return new BestSplitResult(null, 0, null, null);
        }
        int numFeatures = data.get(0).getFeatureCount();

        // loop 1: going through each feature (e.g., Age, CGPA, etc.).
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            // finding all unique values for this feature to use as potential split points.
            Set<Double> uniqueValues = new HashSet<>();
            for (DataPoint dp : data) {
                uniqueValues.add(dp.getFeature(featureIndex));
            }

            // loop 2: going through each unique value to create a question.
            for (double value : uniqueValues) {
                SplitCondition condition = new SplitCondition(featureIndex, value);

                // partitioning the data based on the current question.
                List<DataPoint> leftData = new ArrayList<>();
                List<DataPoint> rightData = new ArrayList<>();
                for (DataPoint dp : data) {
                    if (condition.matches(dp.getFeatures())) {
                        leftData.add(dp);
                    } else {
                        rightData.add(dp);
                    }
                }

                // not considering splits that don't actually divide the data.
                if (leftData.isEmpty() || rightData.isEmpty()) {
                    continue;
                }

                // calculating the weighted average impurity of the two new groups.
                double pLeft = (double) leftData.size() / data.size();
                double weightedGini = pLeft * calculateGini(leftData) + (1.0 - pLeft) * calculateGini(rightData);

                // the info gain
                double informationGain = parentGini - weightedGini;

                // if this split is the best one seen so far, saving it.
                if (informationGain > bestGain) {
                    bestGain = informationGain;
                    bestCondition = condition;
                    bestLeftData = leftData;
                    bestRightData = rightData;
                }
            }
        }
        // returning all the best split components
        return new BestSplitResult(bestCondition, bestGain, bestLeftData, bestRightData);
    }
    // finding the purity of data
    private boolean isPure(List<DataPoint> data) {
        if (data.size() <= 1) {
            return true;
        }
        int firstLabel = data.get(0).getLabel();
        for (int i = 1; i < data.size(); i++) {
            if (data.get(i).getLabel() != firstLabel) {
                return false;
            }
        }
        return true;
    }
}