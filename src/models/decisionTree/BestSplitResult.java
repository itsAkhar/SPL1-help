package models.decisionTree;

import data.DataPoint;
import java.util.List;

// a class to hold all the components of best possible split
public class BestSplitResult {

    // the question with 2 components -> feature and value
    private final SplitCondition condition;

    // the info gain value
    private final double gain;

    // the subset of data that matched the condition (the "yes" group).
    private final List<DataPoint> leftData;

    // the subset of data that did not match the condition (the "no" group).
    private final List<DataPoint> rightData;

    // the constructor
    public BestSplitResult(SplitCondition condition, double gain, List<DataPoint> leftData, List<DataPoint> rightData) {
        this.condition = condition;
        this.gain = gain;
        this.leftData = leftData;
        this.rightData = rightData;
    }

    // getter methods
    public SplitCondition getCondition() { return condition; }
    public double getGain() { return gain; }
    public List<DataPoint> getLeftData() { return leftData; }
    public List<DataPoint> getRightData() { return rightData; }
}