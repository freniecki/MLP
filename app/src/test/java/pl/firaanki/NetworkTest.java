package pl.firaanki;

import junit.framework.TestCase;
import java.util.*;

public class NetworkTest extends TestCase {

    public void testStartNetwork() {

        Network network = new Network(new int[]{4,9,7,3}, -1, 1);
        Map<double[], double[]> data = FileHandler.getFile("iris.data").read();

        ArrayList<Map.Entry<double[], double[]>> dataList = new ArrayList<>(data.entrySet());
        ArrayList<Map.Entry<double[], double[]>> trainData = new ArrayList<>();
        ArrayList<Map.Entry<double[], double[]>> testData = new ArrayList<>();

        Collections.shuffle(dataList);
        int trainCount = 60;

        for (int i = 0; i < trainCount; i++) {
            trainData.add(dataList.get(i));
        }
        for (int i = trainCount; i < 150; i++) {
            testData.add(dataList.get(i));
        }

        network.online(trainData);
        //network.offline(trainData, 0.4);
        network.testNetwork(testData);
    }
}