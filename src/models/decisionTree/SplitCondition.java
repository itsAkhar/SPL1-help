//Concept: A Decision Tree is built on questions. We need a way to represent
// a single question like "Is the student's CGPA less than or equal to 3.0?".
//Components: Every question needs two pieces of information:
//Which feature to look at (CGPA).
//What value to compare against (3.0).
//Outcome: The question always has a binary (yes/no) answer.

package models.decisionTree;

public class SplitCondition {
    // using the feature's index to be general, so I don't have to hardcode "age", "cgpa", etc.
    private final int featureIndex;

    // the value to compare the feature against.
    private final double value;

    // a question is defined by 2 components feature and value
    public SplitCondition(int featureIndex, double value) {
        this.featureIndex = featureIndex;
        this.value = value;
    }

    //  to check if a given student's features match this condition
    public boolean matches(double[] features) {
        // rule is: if the feature value is less than or equal to the split value, it's a "match" -> Yes (go left).
        return features[this.featureIndex] <= this.value;
    }

    // getter methods
    public int getFeatureIndex() {
        return featureIndex;
    }

    public double getValue() {
        return value;
    }
}