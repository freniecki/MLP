package pl.firaanki;

import java.io.Serializable;
import java.util.*;
import java.util.logging.Logger;

public class Network implements Serializable {

    int[] sizes;
    double[][][] weights;
    double[][] biases;
    double[][] activations;
    transient double[][] sums;

    transient Logger logger = Logger.getLogger(getClass().getName());

    String trainStats;
    String testStats;

    Network(int[] sizes, double min, double max) {
        this.sizes = sizes;
        addTrainStats("layers:", arrayToString(sizes));
        weights = Arrays.getWeights(sizes, min, max);
        biases = Arrays.getBias(sizes, min, max);
        activations = new double[sizes.length][];
    }

    public void online(List<Map.Entry<double[], double[]>> trainData) {
        int epochCount = trainData.size();
        for (int epoch = 0; epoch < epochCount; epoch++) {
            double error = 0;
            Collections.shuffle(trainData);
            for (Map.Entry<double[], double[]> current : trainData) {
                double[][][] gradient = countGradientDescent(current.getKey(), current.getValue());

                for (int i = 0; i < sizes.length - 1; i++) {
                    for (int j = 0; j < sizes[i + 1]; j++) {
                        for (int k = 0; k < sizes[i]; k++) {
                            weights[i][j][k] -= gradient[i][j][k];
                            error += gradient[i][j][k];
                        }
                    }
                }
            }
            addTrainStats("training epoch: [" + (epoch + 1) + "]",
                    "|-> error: " + String.format("%.3f", error));
        }
    }

    public void offline(List<Map.Entry<double[], double[]>> trainData, double learningRate) {
        int epochCount = trainData.size();
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

        // sum of gradients in every epoch
        for (int epoch = 0; epoch < epochCount; epoch++) {
            Map.Entry<double[], double[]> current = trainData.get(epoch);
            gradient = countGradientDescent(current.getKey(), current.getValue());

            for (int i = 0; i < sizes.length - 1; i++) {
                for (int j = 0; j < sizes[i + 1]; j++) {
                    for (int k = 0; k < sizes[i]; k++) {
                        bigGradient[i][j][k] += gradient[i][j][k];
                    }
                }
            }
        }

        // update the weights and biases by gradient
        update(epochCount, learningRate, gradientSize, bigGradient);
    }

    private void update(int epochCount, double learningRate, int gradientSize, double[][][] bigGradient) {
        for (int i = 0; i < gradientSize; i++) { // for every layer
            for (int j = 0; j < sizes[i + 1]; j++) { // for every neuron
                for (int k = 0; k < sizes[i]; k++) { // set new weights
                    weights[i][j][k] = weights[i][j][k] - ((learningRate / epochCount) * bigGradient[i][j][k]);
                }
                // set new bias
                biases[i][j] = biases[i][j] - ((learningRate / epochCount) * bigGradient[i][j][sizes[i]]);
            }
        }
    }

    private double countError(Map.Entry<double[], double[]> current) {
        double[][][] gradient = countGradientDescent(current.getKey(), current.getValue());
        double error = 0.0;
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < sizes.length - 1; i++) {
            for (int j = 0; j < sizes[i + 1]; j++) {
                for (int k = 0; k < sizes[i]; k++) {
                    error += gradient[i][j][k];
                }
                if (i == sizes.length - 2) {
                    sb.append(String.format("%.2f", error)).append(" ");
                }
            }
            if (i == sizes.length - 2) {
                addTestStats("last layer errors: " + sb);
            }
        }
        return error;
    }

    public void testNetwork(List<Map.Entry<double[], double[]>> testData) {
        int correct = 0;
        int count = 1;
        for (Map.Entry<double[], double[]> test : testData) {
            statsForTest(test, count);

            if (evaluate(getOutput(), test.getValue())) {
                correct++;
            }
            count++;
        }

        String stats = String.format("%.3f",  (correct / (double) testData.size()));
        addTestStats("test efficiency: " + stats);
    }

    private void statsForTest(Map.Entry<double[], double[]> test, int count) {
        addTestStats("test [" + count + "]:");
        addTestStats("input: " + arrayToString(test.getKey()));

        addTestStats("error: " + countError(test));
        addTestStats("expected output: " + arrayToString(test.getValue()));

        //addTestStats("output values: " + arrayToString(getOutput()));
        for (int i = weights.length - 1; i >= 0; i--) {
            addTestStats("layer activations");
            addTestStats(arrayToString(getActivations(i + 1)));
            addTestStats("layer weights: ");
            for (int j = 0; j < weights[i].length; j++) {
                addTestStats(arrayToString(weights[i][j]));
            }
        }

        addTestStats("-------------------------------");
    }

    private double[] getActivations(int i) {
        return activations[i];
    }

    /*
    wzorca wejściowego,
    popełnionego przez sieć błędu dla całego wzorca,
    pożądanego wzorca odpowiedzi,
    błędów popełnionych na poszczególnych wyjściach sieci,
    wartości wyjściowych neuronów wyjściowych,
    wag neuronów wyjściowych,
    wartości wyjściowych neuronów ukrytych,
    wag neuronów ukrytych (w kolejności warstw od dalszych względem wejść sieci do bliższych)
     */

    private boolean evaluate(double[] output, double[] expected) {
        int outputMax = 0;
        int expectedMax = 0;
        for (int i = 1; i < 3; i++) {
            if (output[i] > output[i - 1]) {
                outputMax = i;
            }
            if (expected[i] > expected[i - 1]) {
                expectedMax = i;
            }
        }
        return outputMax == expectedMax;
    }

    double[] getOutput() {
        int outputIndex = activations.length - 1;
        return activations[outputIndex];
    }

    /**
     * Counts gradient for every wage & bias
     * @param input Training input vector
     * @param output Training expected output
     * @return 3-dimensional gradient array for wages & biases
     */
    private double[][][] countGradientDescent(double[] input, double[] output) {
        countActivations(input);
        int gradientSize = sizes.length - 1; // 1 less than activations

        double[][][] gradient = new double[gradientSize][][];
        double[] deltas = new double[sizes[sizes.length - 1]]; // error for last layer

        // ------------------gradient for output layers---------------

        int lastLayerIndex = gradientSize - 1;
        gradient[lastLayerIndex] = new double[sizes[gradientSize]][];

        for (int i = 0; i < sizes[lastLayerIndex + 1]; i++) { // for every node in last layer
            int wbSize = sizes[lastLayerIndex] + 1; // +1 for bias
            gradient[lastLayerIndex][i] = new double[wbSize];

            double sum = sums[lastLayerIndex][i];
            double delta = (activations[lastLayerIndex + 1][i] - output[i]) * sigmoidDerivative(sum);
            deltas[i] = delta;

            for (int j = 0; j < sizes[lastLayerIndex]; j++) {
                gradient[lastLayerIndex][i][j] = delta * activations[lastLayerIndex][j];
            }

            gradient[lastLayerIndex][i][sizes[lastLayerIndex]] = delta; // bias
        }

        // ---------------gradient for hidden layers------------------

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
        return 1.0 / (1.0 + Math.exp(-v));
    }

    private double sigmoidDerivative(double v) {
        return sigmoid(v) * (1 - sigmoid(v));
    }

    String activationsToString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sizes.length; i++) {
            for (int j = 0; j < activations[i].length; j++) {
                String formatted = String.format("%.2f", activations[i][j]);
                sb.append(formatted).append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    String arrayToString(double[] tab) {
        StringBuilder sb = new StringBuilder();
        for (double d : tab) {
            sb.append(String.format("%.2f",d)).append(" ");
        }
        return sb.toString();
    }

    String arrayToString(int[] tab) {
        StringBuilder sb = new StringBuilder();
        for (int i : tab) {
            sb.append(i).append(" ");
        }
        return sb.toString();
    }

    private void addTrainStats(String line1, String line2) {
        trainStats += line1 + "\n" + line2 + "\n";
    }

    private void addTestStats(String line) {
        testStats += line + "\n";
    }

    public String getTrainStats() {
        return trainStats;
    }

    public String getTestStats() {
        return testStats;
    }
}

