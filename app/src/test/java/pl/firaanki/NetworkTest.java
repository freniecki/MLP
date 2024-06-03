package pl.firaanki;

import junit.framework.TestCase;

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

        System.out.println(network.getStats());

    }

    private ArrayList<Map.Entry<double[], double[]>> getPatterns() {
        Map<double[], double[]> patternsMap = new HashMap<>();
        patternsMap.put(new double[]{1, 0, 0, 0}, new double[]{1, 0, 0, 0});
        patternsMap.put(new double[]{0, 1, 0, 0}, new double[]{0, 1, 0, 0});
        patternsMap.put(new double[]{0, 0, 1, 0}, new double[]{0, 0, 1, 0});
        patternsMap.put(new double[]{0, 0, 0, 1}, new double[]{0, 0, 0, 1});
        return new ArrayList<>(patternsMap.entrySet());
    }

    public void testAutoencoder() {
        ArrayList<Map.Entry<double[], double[]>> patterns = getPatterns();

        Network network = new Network(new int[]{4, 2, 4});
        network.onlineEpoch(patterns, 20, 0.6, 0.0);
        System.out.println(network.getTrainStats());
        network.testNetwork(patterns);
        System.out.println(network.getTestStats());

        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        System.out.println("bias on");

        Network networkBias = new Network(new int[]{4, 2, 4});
        networkBias.setBias();
        networkBias.onlineEpoch(patterns, 20, 0.6, 0.0);
        System.out.println(networkBias.getTrainStats());
        networkBias.testNetwork(patterns);
        System.out.println(networkBias.getTestStats());
    }

    public void research(double learningRate, double momentum) {
        ArrayList<Map.Entry<double[], double[]>> patterns = getPatterns();
        Network network = new Network(new int[]{4, 2, 4});
        network.setBias();
        network.onlineEpoch(patterns, 20, learningRate, momentum);
        System.out.println(network.getTrainStats());
        network.testNetwork(patterns);
        System.out.println(network.getTestStats());
    }

    public void testAutoencoderResearch() {
        research(0.9, 0.0);
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.6, 0.0);
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.2, 0.0);
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.9, 0.6);
        System.out.println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        research(0.2, 0.9);


    }
}