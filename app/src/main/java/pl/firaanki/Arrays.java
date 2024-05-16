package pl.firaanki;

public class Arrays {

    private Arrays() {
    }

    public static double[] getWages(int size, double min, double max) {
        double[] array = new double[size + 1];
        array[0] = 0.0;
        for (int i = 1; i < size + 1; i++) {
            array[i] = min + Math.random() * (max - min);
        }
        return array;
    }

    public static double[][] getLayer(int size, int sizeOfPreviousNeuron, double min, double max) {
        double[][] neuron = new double[size][];
        for (int i = 0; i < size; i++) {
            neuron[i] = getWages(sizeOfPreviousNeuron, min, max);
        }
        return neuron;
    }

    public static double[][][] getNetwork(int[] sizes, double min, double max) {
        double[][][] network = new double[sizes.length][][];
        network[0] = new double[sizes[0]][1];
        for (int i = 1; i < sizes.length; i++) {
            network[i] = getLayer(sizes[i], sizes[i - 1], min, max);
        }
        return network;
    }

    public static double[][][] runEpoch(double[][][] network, double[][] bias) {
        for (int i = 1; i < network.length; i++) { // for every layer
            for (int j = 0; j < network[i].length; j++) { // for every neuron
                network[i][j][0] = sigmoid(countSum(network, i, j)) + bias[i - 1][j];
            }
        }
        return network;
    }

    private static double sigmoid(double v) {
        return 1.0 / (1.0 - Math.exp(-v));
    }


    private static double countSum(double[][][] network, int layer, int neuron) {
        double sum = 0.0;
        for (int i = 0; i < network[layer - 1].length; i++) {
            sum += network[layer - 1][i][0] * network[layer][neuron][i + 1];
        }
        return sum;
    }
}
