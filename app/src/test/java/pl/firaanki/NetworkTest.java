package pl.firaanki;

import junit.framework.TestCase;
import java.util.*;

public class NetworkTest extends TestCase {

    public void testStartNetwork() {

        Network network = new Network(new int[]{4,7,3}, -1, 1);
        Map<double[], double[]> data = FileHandler.getFile("iris.data").read();

        network.startNetwork(6, 30, 0.7, data);

    }
}