package pl.firaanki;

public class Arrays {

    private Arrays() {
    }

    public static double[][][] getWeights(int[] sizes, double min, double max) {
        double[][][] weights = new double[sizes.length - 1][][];

        for (int i = 0; i < sizes.length - 1; i++) {
            weights[i] = new double[sizes[i + 1]][sizes[i]];
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = min + Math.random() * (max - min);
                }
            }
        }

        return weights;
    }

    public static double[][] getBias(int[] sizes, double min, double max) {
        double[][] bias = new double[sizes.length - 1][];

        for (int i = 0; i < sizes.length - 1; i++) {
            bias[i] = new double[sizes[i]];
            for (int j = 0; j < bias[i].length; j++) {
                bias[i][j] = min + Math.random() * (max - min);
            }
        }

        return bias;
    }

    /*
    structure:
    1st layer: [x1], [x2], ..., [xn]
    2nd: [x1, w1, ... wn, b1], ..., [xm, w1, ..., wn, bm]
    ...
     */

    public static double[] getWages(int size, double min, double max) {
        double[] array = new double[size + 2];
        array[0] = 0.0;
        for (int i = 1; i < array.length - 1; i++) {
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



    // graient descent (pl. gradient coś tam)





}
