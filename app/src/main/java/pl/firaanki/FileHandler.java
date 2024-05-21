package pl.firaanki;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.logging.Logger;

public class FileHandler {

    private String fileName;

    Logger logger = Logger.getLogger(getClass().getName());
    private FileHandler(String fileName){this.fileName = fileName;}

    static FileHandler getFile(String fileName) {
        return new FileHandler(fileName);
    }

    public Map<double[], double[]> read() {
        try {
            Scanner myReader = new Scanner(new File(fileName));
            List<String> lines = new ArrayList<>();

            while (myReader.hasNextLine()) {
                String line = myReader.nextLine();
                lines.add(line);
            }

            return changeToDoubleArray(lines);

        } catch (FileNotFoundException e) {
            logger.info("raed exception");
        }
        return null;
    }

    private Map<double[], double[]> changeToDoubleArray(List<String> lines) {
        Map<double[], double[]> data = new HashMap<>();

        for (String s : lines) {
            String[] line = s.split(",");

            // add inputs
            double[] input = new double[4];
            for (int j = 0; j < 4; j++) {
                input[j] = Double.parseDouble(line[j]);
            }

            // add outputs
            double[] output = new double[3];
            switch (line[4]) {
                case "Iris-setosa":
                    output = new double[]{
                            1.0, 0.0, 0.0
                    };
                    break;
                case "Iris-versicolor":
                    output = new double[]{
                            0.0, 1.0, 0.0
                    };
                    break;
                case "Iris-virginica":
                    output = new double[]{
                            0.0, 0.0, 1.0
                    };
                    break;
                default:
                    logger.info("cant set output");
            }
            data.put(input, output);
        }

        return data;
    }
}
