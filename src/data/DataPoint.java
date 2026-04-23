package data;
// contains features , target label and 2 meta feaures
public class DataPoint {


    // storing : [age, gender, cgpa, anxietyValue, depressionValue]
    private final double[] features;


    // storing target stress label (e.g., 0 for "Low", 1 for "Moderate", 2 for "High")
    private final int label;

    // 2 metadata fields , needed for distribution table
    private final int academicYear;  // 0=First, 1=Second, 2=Third, 3=Fourth, 4=Other
    private final int waiver;        // 1=Yes, 0=No

    // the main constructor with features , label and 2 metadata
    public DataPoint(double[] features, int label, int academicYear, int waiver) {
        this.features = features;
        this.label = label;
        this.academicYear = academicYear;
        this.waiver = waiver;
    }

    // anther constructor without 2 metadata
    public DataPoint(double[] features, int label) {
        this(features, label, -1, -1);
    }
    //getter for features[]
    public double[] getFeatures() {
        return features;
    }
    //getter for label
    public int getLabel() {
        return label;
    }

    //getter for specific feature in features[]
    public double getFeature(int index) {
        if (index >= 0 && index < this.features.length) {
            return this.features[index];
        }
        throw new IndexOutOfBoundsException("Feature index " + index + " is out of bounds.");
    }
    //getter for feature count
    public int getFeatureCount() {
        return this.features.length;
    }
    //getter for academic year
    public int getAcademicYear() {
        return academicYear;
    }
    //getter for waiver
    public int getWaiver() {
        return waiver;
    }
}