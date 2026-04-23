package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {

    // here I am converting the age range to a single numerical value (eg,"18-22" -> 20.0)
    private double encodeAge(String ageString) {
        // cleaning the string by removing the quotes("")
        String cleanAgeString = ageString.replace("\"", "").trim();

        // Handling special cases first.
        if (cleanAgeString.equalsIgnoreCase("Below 18")) {
            //I've used 17 as a representative age.
            return 17.0;
        }
        if (cleanAgeString.equalsIgnoreCase("Above 30")) {
            //I've used 31 as a representative age.
            return 31.0;
        }

        // handle the standard "lower-upper" range.
        // using try-catch to handle unexpected input
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

    // female -> 0.0 , male -> 1.0 , prefer not to say -> 2.0
    private double encodeGender(String genderString) {
        // removing the quotes("") and any leading/trailing whitespace.
        String cleanGender = genderString.replace("\"", "").trim();

        // case-insensitive comparison to check the value.
        if (cleanGender.equalsIgnoreCase("Female")) {
            return 0.0;
        } else if (cleanGender.equalsIgnoreCase("Male")) {
            return 1.0;
        } else {
            // "Prefer not to say" or any unexpected values.
            return 2.0;
        }
    }

    // "First Year..." -> 0 , "Second Year..." -> 1 , "Third Year..." -> 2 , "Fourth Year..." -> 3 , "Other" -> 4
    private double encodeAcademicYear(String yearString) {
        // trimming whitespace and quotes
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

    // converting cgpa to a single avg value from their range ( eg: "2.50 - 2.99" -> 2.745 )
    private double encodeCgpa(String cgpaString) {
        String cleanCgpaString = cgpaString.replace("\"", "").trim();

        // handling the "Below" case first
        if (cleanCgpaString.equalsIgnoreCase("Below 2.50")) {
            // I used a representative value, e.g., 2.49
            return 2.49;
        }

        // handle the "Other" case if it exists, otherwise it will be caught by the try-catch
        if (cleanCgpaString.equalsIgnoreCase("Other")) {
            // returning a neutral value 3.0.
            return 3.0;
        }

        // handle the standard "lower - upper" range
        try {
            String[] parts = cleanCgpaString.split("-");
            double lowerBound = Double.parseDouble(parts[0].trim());
            double upperBound = Double.parseDouble(parts[1].trim());
            return (lowerBound + upperBound) / 2.0;
        } catch (Exception e) {
            System.err.println("Could not parse CGPA string: " + cgpaString);
            return 0.0; // returning a default value
        }
    }

    // if yes -> 1.0 , if no -> 0.0
    private double encodeScholarship(String scholarshipString) {
        String cleanString = scholarshipString.replace("\"", "").trim();
        if (cleanString.equalsIgnoreCase("Yes")) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    //"Low Stress" -> 0 , "Moderate Stress" -> 1 , "High Perceived Stress" -> 2
    private int encodeStressLabel(String labelString) {
        String cleanLabel = labelString.replace("\"", "").trim();

        // using a case-insensitive check.
        if (cleanLabel.equalsIgnoreCase("High Perceived Stress")) {
            return 2;
        } else if (cleanLabel.equalsIgnoreCase("Moderate Stress")) {
            return 1;
        } else { // for "Low Stress"
            return 0;
        }
    }
    // this is the loader method which reads the CSV file
    public List<DataPoint> loadData(String filePath) {
        // making an arraylist of dataPoint Class( data structure )
        List<DataPoint> dataPoints = new ArrayList<>();
        String line = "";

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // skip header

            while ((line = br.readLine()) != null) {
                try {
                    //using the smarter split
                    // It uses a regular expression (split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1))
                    // to correctly split CSV lines, even when some fields (like the university name) contain commas within quotes. This prevents data corruption.
                    String[] values = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);

                    // all the features in order
                    double age = encodeAge(values[0]);
                    double gender = encodeGender(values[1]);
                    double academicYear = encodeAcademicYear(values[4]);
                    double cgpa = encodeCgpa(values[5]);
                    double scholarship = encodeScholarship(values[6]);
                    double anxietyValue = Double.parseDouble(values[26].replace("\"", "").trim());
                    double depressionValue = Double.parseDouble(values[37].replace("\"", "").trim());

                    // the 5 main features
                    double[] features = new double[] {
                            age, gender, cgpa, anxietyValue, depressionValue
                    };
                    // my target feature
                    int stressLabel = encodeStressLabel(values[18]);

                    // storing academic year and waiver also for distribution table
                    int ayMeta = (int) academicYear;
                    int waiverMeta = (int) scholarship;

                    dataPoints.add(new DataPoint(features, stressLabel, ayMeta, waiverMeta));

                } catch (Exception e) {
                    System.err.println("Skipping malformed line: " + line + " | Error: " + e.getMessage());
                    // it's helpful to also print which part failed
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataPoints;
    }
}