package pl.firaanki;

import java.util.*;
import java.util.logging.Logger;

public class Network {

    int[] sizes;
    double[][][] weights;
    double[][] biases;
    double[][] activations;
    double[][] sums;

    Logger logger = Logger.getLogger(getClass().getName());

    Network(int[] sizes, double min, double max) {
        this.sizes = sizes;
        weights = Arrays.getWeights(sizes, min, max);
        biases = Arrays.getBias(sizes, min, max);
        activations = new double[sizes.length][];
    }

    public void startNetwork(int epochCount, int testCount, double learningRate, Map<double[], double[]> data) {
        int gradientSize = sizes.length - 1;
        double[][][] bigGradient = new double[gradientSize][][];
        double[][][] gradient;

        // initialize the network
        for (int i = 0; i < gradientSize; i++) {
            bigGradient[i] = new double[sizes[i + 1]][];
            for (int j = 0; j < sizes[i + 1]; j++) {
                // create space for weights & bias
                bigGradient[i][j] = new double[sizes[i] + 1];
            }
        }

        ArrayList<Map.Entry<double[], double[]>> dataList = new ArrayList<>(data.entrySet());
        ArrayList<Map.Entry<double[], double[]>> trainData = new ArrayList<>();
        ArrayList<Map.Entry<double[], double[]>> testData = new ArrayList<>();

        for (int i = 0; i < testCount; i++) {
            trainData.add(dataList.get(i));
        }
        for (int i = testCount; i < 150; i++) {
            testData.add(dataList.get(i));
        }



        // sum of gradients in every epoch
        for (int epoch = 0; epoch < epochCount; epoch++) {
            Map.Entry<double[], double[]> current = trainData.get(epoch);
            gradient = countGradientDescent(current.getKey(), current.getValue());
            String info = "-----epoch " + epoch + " -----";
            logger.info(info);
            displayActivations();
            for (int i = 0; i < sizes.length - 1; i++) {
                for (int j = 0; j < sizes[i + 1]; j++) {
                    for (int k = 0; k < sizes[i]; k++) {
                        bigGradient[i][j][k] += gradient[i][j][k];
                    }
                }
            }
        }

        // update the weights and biases by gradient
        for (int i = 0; i < gradientSize; i++) { // for every layer
            for (int j = 0; j < sizes[i + 1]; j++) { // for every neuron
                for (int k = 0; k < sizes[i]; k++) { // set new weights
                    weights[i][j][k] = (weights[i][j][k] - (learningRate / epochCount)) * bigGradient[i][j][k];
                }
                // set new bias
                biases[i][j] = (biases[i][j] - (learningRate / epochCount)) * bigGradient[i][j][sizes[i]];
            }
        }

        Map.Entry<double[], double[]> test1 = testData.getFirst();
        countActivations(test1.getKey());

        displayDoubleTab(test1.getValue());
        logger.info(arrayToString(getOutput()));
    }

    double[] getOutput() {
        int outputIndex = activations.length - 1;
        return activations[outputIndex];
    }

    String arrayToString(double[] tab) {
        StringBuilder sb = new StringBuilder();
        for (double d : tab) {
            sb.append(d + " ");
        }
        return sb.toString();
    }

    /**
     * Counts gradient for every wage & bias
     * @param input Training input vector
     * @param output Training expected output
     * @return 3-dimensional gradient array for wages & biases
     */
    private double[][][] countGradientDescent(double[] input, double[] output) {
        countActivations(input);
        double[][][] gradient = new double[sizes.length - 1][][];
        double[] deltas = new double[sizes[sizes.length - 1]]; // błąd dla ostatniej warstwy

        // ---------------------------------
        // Obliczanie gradientu dla warstwy wyjściowej

        int lastLayerIndex = sizes.length - 2;
        gradient[lastLayerIndex] = new double[sizes[lastLayerIndex + 1]][];

        for (int i = 0; i < sizes[lastLayerIndex + 1]; i++) { // dla każdego neuronu w ostatniej warstwie
            gradient[lastLayerIndex][i] = new double[sizes[lastLayerIndex] + 1]; // +1 dla biasu

            double sum = sums[lastLayerIndex][i];
            double delta = (activations[lastLayerIndex + 1][i] - output[i]) * sigmoidDerivative(sum);
            deltas[i] = delta;

            for (int j = 0; j < sizes[lastLayerIndex]; j++) {
                gradient[lastLayerIndex][i][j] = delta * activations[lastLayerIndex][j];
            }

            gradient[lastLayerIndex][i][sizes[lastLayerIndex]] = delta; // bias
        }

        // ---------------------------------
        // Obliczanie gradientu dla warstw ukrytych

        for (int i = lastLayerIndex - 1; i >= 0; i--) {
            int neurons = sizes[i + 1];
            double[] nextDeltas = new double[neurons];
            gradient[i] = new double[neurons][];

            for (int j = 0; j < neurons; j++) {
                gradient[i][j] = new double[sizes[i] + 1]; // +1 dla biasu

                double deltaSum = 0.0;
                for (int k = 0; k < sizes[i + 2]; k++) {
                    deltaSum += weights[i + 1][k][j] * deltas[k];
                }

                double delta = deltaSum * sigmoidDerivative(sums[i][j]);
                nextDeltas[j] = delta;

                for (int k = 0; k < sizes[i]; k++) {
                    gradient[i][j][k] = delta * activations[i][k];
                }

                gradient[i][j][sizes[i]] = delta; // bias
            }

            deltas = nextDeltas;
        }

        return gradient;
    }

    private void countActivations(double[] input) {
        sums = new double[sizes.length - 1][];

        activations[0] = new double[sizes[0]];
        System.arraycopy(input, 0, activations[0], 0, sizes[0]);

        double sum = 0.0;

        for (int i = 1; i < sizes.length; i++) { // for every layer but 1st
            activations[i] = new double[sizes[i]]; // creates space for activations in layer

            if (i == 0) {

            }

            sums[i - 1] = new double[sizes[i]]; // holds sums for every activation

            for (int j = 0; j < sizes[i]; j++) { // for every neuron in layer
                for (int k = 0; k < sizes[i - 1]; k++) { // for all wages in neuron
                    // activation of previous layer * wage for that neuron
                    sum += activations[i - 1][k] * weights[i - 1][j][k];
                }
                // counts activation value for neuron
                activations[i][j] = sigmoid(sum) + biases[i - 1][j];
                // holds sum value for neuron
                sums[i - 1][j] = sum;

                sum = 0.0;
            }
        }
    }

    private double sigmoid(double v) {
        return 1.0 / (1.0 - Math.exp(-v));
    }

    private double sigmoidDerivative(double v) {
        return sigmoid(v) * (1 - sigmoid(v));
    }

    private void displayActivations() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sizes.length; i++) {
            for (int j = 0; j < activations[i].length; j++) {
                sb.append(activations[i][j]).append(" ");
            }
            sb.append("\n");
        }
        logger.info(sb.toString());
    }

    private void displayDoubleTab(double[] tab) {
        StringBuilder sb = new StringBuilder();
        for (double v : tab) {
            sb.append(v).append(" ");
        }
        sb.append("\n");
        logger.info(sb.toString());
    }
}

