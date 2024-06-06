package pl.firaanki;

import junit.framework.TestCase;

import javax.swing.*;
import java.util.*;

public class NetworkTest extends TestCase {

    public void testStartNetwork() {

        Network network = new Network(new int[]{4, 9, 7, 3});
        Map<double[], double[]> data = FileHandler.getFile("iris.data").read();

        ArrayList<Map.Entry<double[], double[]>> dataList = new ArrayList<>(data.entrySet());
        ArrayList<Map.Entry<double[], double[]>> trainData = new ArrayList<>();
        ArrayList<Map.Entry<double[], double[]>> testData = new ArrayList<>();

        int trainCount = 60;

        for (int i = 0; i < trainCount; i++) {
            trainData.add(dataList.get(i));
        }
        for (int i = trainCount; i < 150; i++) {
            testData.add(dataList.get(i));
        }

        network.onlineEpoch(trainData, 20, 1, 0);
        System.out.println(network.getTrainStats());
        FileHandler.getFile("train.txt").write(network.getTrainStats());

        network.testNetwork(testData);
        System.out.println(network.getTestStats());
        FileHandler.getFile("test.txt").write(network.getTestStats());

        int[][] stats = network.getOutputStats();
        Statistics statistics = new Statistics(stats);
        System.out.println(statistics.getAllStats());
    }

    public static void main(String[] args) {
        checkBias(true, 2, "with bias");
        checkBias(false, 2, "without bias");
    }

    static void checkBias(boolean bias, int hiddenNeurons, String title) {
        ArrayList<Map.Entry<double[], double[]>> patterns = getPatterns();

        Network network = new Network(new int[]{4, hiddenNeurons, 4});
        if (bias) {
            network.setBias();
        }

        network.onlineEpoch(patterns, 1000, 0.6, 0.0);
        network.testNetwork(patterns);

        System.out.println(network.getTestStats());

        plot(network.getErrors(), title);
    }

    public void testAutoencoderResearch() {
        research(0.9, 0.0, "lr: 0.9, m: 0.0");
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.6, 0.0, "lr: 0.6, m: 0.0");
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.2, 0.0, "lr: 0.2, m: 0.0");
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.9, 0.6,"lr: 0.9, m: 0.6");
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.2, 0.9, "lr: 0.2, m: 0.9");
    }

    static void plot(double[] errors, String title) {
        SwingUtilities.invokeLater(() -> {
            Plot example = new Plot(title, errors);
            example.setSize(800, 400);
            example.setLocationRelativeTo(null);
            example.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            example.setVisible(true);
        });
    }

    public void research(double learningRate, double momentum, String title) {
        ArrayList<Map.Entry<double[], double[]>> patterns = getPatterns();
        Network network = new Network(new int[]{4, 2, 4});
        network.setBias();
        network.onlineEpoch(patterns, 20, learningRate, momentum);
        network.testNetwork(patterns);
        System.out.println(network.getTestStats());
        plot(network.getErrors(), title);
    }

    private static ArrayList<Map.Entry<double[], double[]>> getPatterns() {
        Map<double[], double[]> patternsMap = new HashMap<>();
        patternsMap.put(new double[]{1, 0, 0, 0}, new double[]{1, 0, 0, 0});
        patternsMap.put(new double[]{0, 1, 0, 0}, new double[]{0, 1, 0, 0});
        patternsMap.put(new double[]{0, 0, 1, 0}, new double[]{0, 0, 1, 0});
        patternsMap.put(new double[]{0, 0, 0, 1}, new double[]{0, 0, 0, 1});
        return new ArrayList<>(patternsMap.entrySet());
    }
}