package pl.firaanki;

import junit.framework.TestCase;

public class NetworkTest extends TestCase {

    public void testStartNetwork() {

        Network network = new Network(new int[]{4,7,3}, -1, 1);

        double[][] inputs = {
                {1.0, 0.2, 0.2, 0.2},
                {1.0, 0.3, 0.2, 0.1},
                {0.2, 1.0, 1.0, 0.1},
                {0.3, 1.0, 1.0, 0.2},
                {0.2, 0.1, 0.3, 1.0},
                {0.1, 0.2, 0.1, 1.0}
        };

        double[][] outputs = {
                {1.0, 0.0, 0.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0},
                {0.0, 0.0, 1.0}
        };

        network.startNetwork(6, 0.7, inputs, outputs);

    }
}