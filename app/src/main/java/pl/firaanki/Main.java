package pl.firaanki;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Logger;

public class Main {

    static Logger logger = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        logger.info("ile warstw");
        int size = scanner.nextInt();

        int[] sizes = new int[size];

        logger.info("ile neuronów w warstwie?");
        for (int i = 0; i < size; i++) {
            logger.info((i + 1) + ": ");
            sizes[i] = scanner.nextInt();
        }

        logger.info("uwzględnić bias? [t/n]");
        String bias = scanner.nextLine();
        boolean takeBias = bias.equalsIgnoreCase("t");

        logger.info("podaj wspolczynnik nauki");
        double learningRate = scanner.nextDouble();

        logger.info("podaj momentum");
        double momentum = scanner.nextDouble();

        logger.info("podaj liczbe epok");
        int epochs = scanner.nextInt();

        logger.info("podaj rozmiar zbioru treningowego");
        int trainCount = scanner.nextInt();

        Map<double[], double[]> data = FileHandler.getFile("iris.data").read();
        ArrayList<Map.Entry<double[], double[]>> dataList = new ArrayList<>(data.entrySet());
        ArrayList<Map.Entry<double[], double[]>> trainData = new ArrayList<>();
        ArrayList<Map.Entry<double[], double[]>> testData = new ArrayList<>();

        for (int i = 0; i < trainCount; i++) {
            trainData.add(dataList.get(i));
        }
        for (int i = trainCount; i < 150; i++) {
            testData.add(dataList.get(i));
        }
        Network network = new Network(sizes);
        if (takeBias) {
            network.setBias();
        }

        network.onlineEpoch(trainData, epochs, learningRate, momentum);
        network.testNetwork(testData);

        plot(network);

    }

    static void plot(Network network) {
        SwingUtilities.invokeLater(() -> {
            Plot example = new Plot("Error Plot Example", network.getErrors());
            example.setSize(800, 400);
            example.setLocationRelativeTo(null);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setVisible(true);
        });
    }
}
