package models.decisionTree;

public class Node {

    // --- Fields for a Decision Node ---
    private final SplitCondition splitCondition; // the question to ask
    private final Node leftChild;              // the "Yes" branch
    private final Node rightChild;             // the "No" branch

    // --- Field for a Leaf Node ---
    private final int prediction;              // the final answer

    // constructor for a decision node
    public Node(SplitCondition splitCondition, Node leftChild, Node rightChild) {
        this.splitCondition = splitCondition;
        this.leftChild = leftChild;
        this.rightChild = rightChild;
        this.prediction = -1; // -1 to show this is not a leaf
    }

    // constructor for a leaf node
    public Node(int prediction) {
        this.prediction = prediction;
        this.splitCondition = null; // no ques
        this.leftChild = null;      // no children
        this.rightChild = null;
    }

    // helper function to check a node is leaf or not
    public boolean isLeaf() {
        return this.leftChild == null && this.rightChild == null;
    }

    // getter methods for node fields
    public SplitCondition getSplitCondition() { return splitCondition; }
    public Node getLeftChild() { return leftChild; }
    public Node getRightChild() { return rightChild; }
    public int getPrediction() { return prediction; }
}
