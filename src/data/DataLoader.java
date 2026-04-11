package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataLoader {
    /**
     * Converts the age string from the CSV into a single numerical value.
     * e.g., "18-22" -> 20.0
     * @param ageString The string from the CSV, which may include quotes.
     * @return A double representing the age.
     */
    private double encodeAge(String ageString) {
        // 1. Clean the string by removing quotes.
        String cleanAgeString = ageString.replace("\"", "").trim();

        // 2. Handle special cases first.
        if (cleanAgeString.equalsIgnoreCase("Below 18")) {
            //I've used 17 as a representative age.
            return 17.0;
        }
        if (cleanAgeString.equalsIgnoreCase("Above 30")) {
            //I've used 31 as a representative age.
            return 31.0;
        }

        // 3. Handle the standard "lower-upper" range.
        // Use a try-catch block for safety in case the format is unexpected.
        try {
            String[] parts = cleanAgeString.split("-");
            double lowerBound = Double.parseDouble(parts[0].trim());
            double upperBound = Double.parseDouble(parts[1].trim());
            return (lowerBound + upperBound) / 2.0;
        } catch (NumberFormatException e) {
            // If something goes wrong, print an error and return a default value like 0.
            System.err.println("Could not parse age string: " + ageString);
            return 0.0;
        }
    }

    /**
     * Converts the gender string from the CSV into a numerical code.
     * "Female" -> 0.0
     * "Male" -> 1.0
     * "Prefer not to say" (and any other value) -> 2.0
     * @param genderString The string from the CSV, e.g., "\"Female\"".
     * @return A double representing the encoded gender.
     */
    private double encodeGender(String genderString) {
        // First, remove the quotes and any leading/trailing whitespace.
        String cleanGender = genderString.replace("\"", "").trim();

        // Use of a case-insensitive comparison to check the value.
        if (cleanGender.equalsIgnoreCase("Female")) {
            return 0.0;
        } else if (cleanGender.equalsIgnoreCase("Male")) {
            return 1.0;
        } else {
            // This will handle "Prefer not to say" or any unexpected values.
            return 2.0;
        }
    }

    /**
     * Converts the academic year string into a numerical code.
     * "First Year..." -> 0
     * "Second Year..." -> 1
     * "Third Year..." -> 2
     * "Fourth Year..." -> 3
     * "Other" -> 4
     * @param yearString The string from the CSV.
     * @return An integer code for the academic year.
     */
    private double encodeAcademicYear(String yearString) {
        String cleanYearString = yearString.replace("\"", "").trim();

        if (cleanYearString.contains("First Year")) {
            return 0.0;
        } else if (cleanYearString.contains("Second Year")) {
            return 1.0;
        } else if (cleanYearString.contains("Third Year")) {
            return 2.0;
        } else if (cleanYearString.contains("Fourth Year")) {
            return 3.0;
        } else { // "Other"
            return 4.0;
        }
    }
    /**
     * Converts the CGPA string from the CSV into a single numerical value which is their avg.
     * e.g., "2.50 - 2.99" -> 2.745
     * @param cgpaString The string from the CSV.
     * @return A double representing the CGPA.
     */
    private double encodeCgpa(String cgpaString) {
        String cleanCgpaString = cgpaString.replace("\"", "").trim();

        // Handle the "Below" case first
        if (cleanCgpaString.equalsIgnoreCase("Below 2.50")) {
            // I used a representative value, e.g., 2.49
            return 2.49;
        }

        // Handle the "Other" case if it exists, otherwise it will be caught by the try-catch
        if (cleanCgpaString.equalsIgnoreCase("Other")) {
            // Returning a neutral value like 3.0 might be okay.
            return 3.0;
        }

        // Handle the standard "lower - upper" range
        try {
            String[] parts = cleanCgpaString.split("-");
            double lowerBound = Double.parseDouble(parts[0].trim());
            double upperBound = Double.parseDouble(parts[1].trim());
            return (lowerBound + upperBound) / 2.0;
        } catch (Exception e) {
            // If parsing fails for any reason
            System.err.println("Could not parse CGPA string: " + cgpaString);
            return 0.0; // Return a default value
        }
    }
    /**
     * Converts the waiver/scholarship status into a numerical code.
     * "Yes" -> 1.0
     * "No" -> 0.0
     * @param scholarshipString The string from the CSV.
     * @return 1.0 if "Yes", 0.0 otherwise.
     */
    private double encodeScholarship(String scholarshipString) {
        String cleanString = scholarshipString.replace("\"", "").trim();
        if (cleanString.equalsIgnoreCase("Yes")) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    /**
     * Converts the stress label string into the target integer class.
     * "Low Stress" -> 0
     * "Moderate Stress" -> 1
     * "High Perceived Stress" -> 2
     * @param labelString The string from the CSV, e.g., "\"High Perceived Stress\"".
     * @return An integer representing the class (0, 1, or 2).
     */
    private int encodeStressLabel(String labelString) {
        // Firstly, removed the quotes and any leading/trailing whitespace.
        String cleanLabel = labelString.replace("\"", "").trim();

        // Use a case-insensitive check.
        // We check for "High" first as it's a more specific substring.
        if (cleanLabel.equalsIgnoreCase("High Perceived Stress")) {
            return 2;
        } else if (cleanLabel.equalsIgnoreCase("Moderate Stress")) {
            return 1;
        } else { // This will catch "Low Stress"
            return 0;
        }
    }

    public List<DataPoint> loadData(String filePath) {
        List<DataPoint> dataPoints = new ArrayList<>();
        String line = "";

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // Skip header

            while ((line = br.readLine()) != null) {
                try {
                    //USING THE SMARTER SPLIT
                    String[] values = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);

                    // all the features in order
                    double age = encodeAge(values[0]);
                    double gender = encodeGender(values[1]);
                    double academicYear = encodeAcademicYear(values[4]);
                    double cgpa = encodeCgpa(values[5]);
                    double scholarship = encodeScholarship(values[6]);
                    double anxietyValue = Double.parseDouble(values[26].replace("\"", "").trim());
                    double depressionValue = Double.parseDouble(values[37].replace("\"", "").trim());

                    double[] features = new double[] {
                            age, gender, cgpa, anxietyValue, depressionValue
                    };

                    int stressLabel = encodeStressLabel(values[18]);

                    dataPoints.add(new DataPoint(features, stressLabel));

                } catch (Exception e) {
                    System.err.println("Skipping malformed line: " + line + " | Error: " + e.getMessage());
                    // It's helpful to also print which part failed
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataPoints;
    }

//    public static void main(String[] args) {
//        // the csv file
//        String filePath = "MentalHealth.csv";
//
//        DataLoader loader = new DataLoader();
//        List<DataPoint> data = loader.loadData(filePath);
//
//        System.out.println("Successfully loaded " + data.size() + " data points.");
//
//        if (data.size() > 0) {
//            System.out.println("\n--- Verifying first 3 data points ---");
//
//            // Print the first 3 to manually check them
//            for (int i = 0; i < 3 && i < data.size(); i++) {
//                DataPoint dp = data.get(i);
//                System.out.println("\nDataPoint " + (i + 1) + ":");
//                System.out.println("  Features: " + Arrays.toString(dp.getFeatures()));
//                System.out.println("  Label: " + dp.getLabel());
//            }
////            System.out.println("  Features: " + Arrays.toString(data.get(1622).getFeatures()));
////            System.out.println("  Label: " + data.get(1622).getLabel());
//
//            System.out.println("\nExpected for first DataPoint (from your CSV):");
//            System.out.println("  [20.0, 0.0, 1.0, 2.745, 0.0, 15.0, 20.0]"); // Age, Gender, Acad.Year, CGPA, Scholarship, Anxiety, Depression
//            System.out.println("  Label: 2 (High)");
//        }
//    }

}
