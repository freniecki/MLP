package pl.firaanki;

public class Network {

    int[] sizes;
    double[][][] weights;
    double[][] biases;
    double[][] activations;
    double[][] sums;

    Network(int[] sizes, double min, double max) {
        this.sizes = sizes;
        weights = Arrays.getWeights(sizes, min, max);
        biases = Arrays.getBias(sizes, min, max);
        activations = new double[sizes.length - 1][];
    }

    public void startNetwork(int epochCount, double learningRate, double[][] inputs, double[][] outputs) {
        double[][][] bigGradient = new double[sizes.length - 2][][];
        double[][][] gradient;

        for (int epoch = 0; epoch < epochCount; epoch++) {
            gradient = countGradientDescent(inputs[epoch], outputs[epoch]);
            for (int i = 0; i < sizes.length - 1; i++) {
                for (int j = 0; j < sizes[i + 1]; j++) {
                    for (int k = 0; k < sizes[i]; k++) {
                        bigGradient[i][j][k] += gradient[i][j][k];
                    }
                }
            }
        }

        for (int i = 0; i < sizes.length - 2; i++) { // for every layer
            for (int j = 0; j < sizes[i + 1]; j++) { // for every neuron
                for (int k = 0; k < sizes[i]; k++) { // set new weights
                    weights[i][j][k] = (weights[i][j][k] - (learningRate / epochCount)) * bigGradient[i][j][k];
                }
                // set new bias
                biases[i][j] = (biases[i][j] - (learningRate / epochCount)) * bigGradient[i][j][sizes[i]];
            }
        }
    }

    /**
     * Counts gradient for every wage & bias
     * @param input Training input vector
     * @param output Training expected output
     * @return 3-dimensional gradient array for wages & biases
     */
    private double[][][] countGradientDescent(double[] input, double[] output) {
        countActivations(input);
        double[][][] gradient = new double[sizes.length - 2][][];
        double delta; // error for given neuron

        // ---------------------------------
        // count gradient for output layer

        int lastLayerIndex = sizes.length - 1;
        gradient[lastLayerIndex] = new double[sizes[lastLayerIndex]][];
        for (int i = 0; i < sizes[lastLayerIndex]; i++) { // for every neuron in last layer
            // create space for wages & bias
            gradient[lastLayerIndex][i] = new double[sizes[lastLayerIndex - 1]];

            // count delta for every activation
            delta = (activations[lastLayerIndex][i] - output[i]) * sigmoidDerivative(sums[sums.length - 1][i]);

            // for every wage in neuron
            for (int j = 0; j < sizes[lastLayerIndex - 1] - 1; j++) { // set every wage
                gradient[gradient.length - 1][i][j] = delta * activations[lastLayerIndex - 1][j];
            }

            // set bias
            gradient[sizes[lastLayerIndex - 1]][i][sizes[lastLayerIndex - 2]] = delta;
        }

        // ---------------------------------
        // count gradient for hidden layers

        for (int i = sizes.length - 2; i >= 1; i--) {
            gradient[i] = new double[sizes[i]][]; // holds gradient for every wage&bias in layer

            for (int j = 0; j <= sizes[i]; j++) { // for every neuron
                // create space for wages & bias
                gradient[i][j] = new double[sizes[i - 1] + 1];

                delta = (activations[i][j] - output[j]) * sigmoidDerivative(sums[i - 1][j]);

                for (int k = 1; k < sizes[i - 1]; k++) { // update every wage
                    // delta * activation of previous layer
                    gradient[i][j][k] = delta * activations[i - 1][k - 1];
                }

                gradient[i][j][sizes[i - 1]] = delta; // update bias
            }
        }

        return gradient;
    }

    private void countActivations(double[] input) {
        sums = new double[sizes.length - 2][];

        if (sizes[0] >= 0) System.arraycopy(input, 0, activations[0], 0, sizes[0]);

        double sum = 0.0;

        for (int i = 1; i < sizes.length; i++) { // for every layer but 1st
            sums[i - 1] = new double[sizes[i]]; // holds sums for every activation

            for (int j = 0; j < sizes[i]; j++) { // for every neuron in layer
                for (int k = 0; k < sizes[i - 1]; k++) { // for all wages in neuron
                    // activation of previous layer * wage for that neuron
                    sum += activations[i - 1][k] * weights[i][j][k];
                }
                // counts activation value for neuron
                activations[i][j] = sigmoid(sum) + biases[i][j];
                // holds sum value for neuron
                sums[i][j] = sum;

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
}

