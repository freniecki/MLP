package pl.firaanki;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.logging.Logger;

public class FileHandler {

    private final String fileName;

    Logger logger = Logger.getLogger(getClass().getName());

    private FileHandler(String fileName) {
        this.fileName = fileName;
    }

    public static FileHandler getFile(String fileName) {
        return new FileHandler(fileName);
    }

    public Map<double[], double[]> read() {
        StringBuilder sb = new StringBuilder();

        Path path = new File(fileName).toPath().toAbsolutePath();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            int character;
            while ((character = reader.read()) != -1) {
                sb.append((char) character);
            }
        } catch (IOException e) {
            logger.info("raed exception");
        }

        String longLine = sb.toString();
        List<String> lines;
        if (System.getProperty("os.name").contains("Windows")) {
            lines = List.of(longLine.split("\r\n"));
        }
        else {
            lines = List.of(longLine.split("\n"));
        }

        return scale(format(lines));
    }

    private Map<double[],double[]> scale(Map<double[],double[]> format) {
        Map<double[],double[]> scaled = new HashMap<>();
        double[] max = new double[]{0,0,0,0};
        double[] min = new double[]{Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE};
        for (Map.Entry<double[],double[]> entry : format.entrySet()) {
            double[] current = entry.getKey();
            for (int i=0;i<4;i++) {
                double curr = current[i];
                if (curr > max[i]) {
                    max[i] = curr;
                }
                if (curr < min[i]) {
                    min[i] = curr;
                }
            }
        }

        for (Map.Entry<double[],double[]> entry : format.entrySet()) {
            double[] current = entry.getKey();
            for (int i = 0; i < 4; i++) {
                current[i] = (current[i] - min[i]) / (max[i] - min[i]);
            }
            scaled.put(current, entry.getValue());
        }

        return scaled;
    }

    private Map<double[], double[]> format(List<String> lines) {
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
