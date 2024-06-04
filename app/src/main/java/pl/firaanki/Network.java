package pl.firaanki;

import java.io.Serializable;
import java.util.*;

public class Network implements Serializable {

    /**
     * Array defining layers' sizes in MLP.
     * n - number of layers
     */
    int[] sizes;
    /**
     * n - 1 size
     */
    double[][][] weights;
    /**
     * n - 1 size
     */
    double[][] activations;

    double[][][] velocityWeights;
    double[][] velocityBiases;
    transient double[][] sums;

    static final int MIN = -1;
    static final int MAX = 1;

    private transient String trainStats = "";
    private transient String testStats = "";
    private final int[][] outputStats;
    private double[] errors;

    boolean takeBias = false;
    boolean doShuffle = true;
    boolean[] sizesStats;

    Network(int[] sizes) {
        this.sizes = sizes;
        sizesStats = new boolean[sizes.length];
        writeTrainLine("Layers structure:");
        writeTrainLine(arrayToString(sizes));

        weights = ArrayLib.getWeights(sizes, MIN, MAX);
        activations = new double[sizes.length][];

        velocityWeights = new double[sizes.length - 1][][];
        velocityBiases = new double[sizes.length - 1][];
        for (int i = 0; i < sizes.length - 1; i++) {
            velocityWeights[i] = new double[sizes[i + 1]][sizes[i]];
            velocityBiases[i] = new double[sizes[i + 1]];
        }

        outputStats = new int[sizes[sizes.length - 1]][3];
    }

    void setBias() {
        takeBias = true;
    }
    void turnOffShuffle() {
        doShuffle = false;
    }
    void setSizesStats(boolean[] sizesStats) {
        this.sizesStats = sizesStats;
    }

    /*--------------------------------------MLP operations--------------------------------------*/

    public void onlineEpoch(List<Map.Entry<double[], double[]>> trainData, int epochs,
                            double learningRate, double momentum) {
        errors = new double[epochs];
        writeTrainLine("--------------TRAINING--------------");
        for (int epoch = 0; epoch < epochs; epoch++) {
            double error = 0;
            error = doEpoch(trainData, learningRate, momentum, error);
            errors[epoch] = error;

            writeTrainLine("epoch no." + (epoch + 1) + " | error: " + String.format("%.3f", error));
        }
    }

    public void onlinePrecise(List<Map.Entry<double[], double[]>> trainData, double precision,
                              double learningRate, double momentum) {
        double error = Double.MAX_VALUE;
        int epoch = 0;

        while (error > precision || epoch > 50) {
            error = doEpoch(trainData, learningRate, momentum, error);

            writeTrainLine("Training epoch no." + (++epoch));
            writeTrainLine("|-> error: " + String.format("%.3f", error));
        }
    }

    private double doEpoch(List<Map.Entry<double[], double[]>> trainData, double learningRate, double momentum, double error) {
        if (doShuffle) {
            Collections.shuffle(trainData);
        }
        for (Map.Entry<double[], double[]> current : trainData) {
            countActivations(current.getKey());
            double[][][] gradient = countGradientDescent(current.getValue());

            error = updateOnline(learningRate, momentum, gradient, error);
        }
        return error;
    }

    private double updateOnline(double learningRate, double momentum, double[][][] gradient, double error) {
        for (int i = 0; i < sizes.length - 1; i++) {
            for (int j = 0; j < sizes[i + 1]; j++) {
                for (int k = 0; k < sizes[i]; k++) {
                    velocityWeights[i][j][k] = momentum * velocityWeights[i][j][k] + learningRate * gradient[i][j][k];
                    weights[i][j][k] -= velocityWeights[i][j][k];
                    error += gradient[i][j][k];
                }
                if (takeBias) {
                    velocityBiases[i][j] = momentum * velocityBiases[i][j] + learningRate * gradient[i][j][sizes[i]];
                    weights[i][j][sizes[i]] -= velocityBiases[i][j];
                    error += gradient[i][j][sizes[i]];
                }
            }
        }
        return error;
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
                if (takeBias) {
                    sum += 1 * weights[i - 1][j][sizes[i - 1]]; // count bias input
                }
                // counts activation value for neuron
                activations[i][j] = sigmoid(sum);
                // holds sum value for neuron
                sums[i - 1][j] = sum;

                sum = 0.0;
            }
        }
    }

    private double[][][] countGradientDescent(double[] output) {
        int gradientSize = sizes.length - 1; // 1 less than activations

        double[][][] gradient = new double[gradientSize][][];
        double[] deltas = new double[sizes[sizes.length - 1]]; // error for last layer

        // ------------------gradient for output layers---------------

        int lastLayerIndex = gradientSize - 1;
        gradient[lastLayerIndex] = new double[sizes[gradientSize]][];

        for (int i = 0; i < sizes[lastLayerIndex + 1]; i++) { // for every neuron in last layer
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

    /*--------------------------------------Test operations--------------------------------------*/

    public void testNetwork(List<Map.Entry<double[], double[]>> testData) {
        int correct = 0;
        int count = 1;
        writeTestLine("--------------TESTING--------------");

        for (Map.Entry<double[], double[]> test : testData) {
            countActivations(test.getKey());
            statsForTest(test, count);

            double[] scaledOutput = getScaledOutput();
            double[] realOutput = test.getValue();

            if (Arrays.equals(scaledOutput, realOutput)) {
                correct++;
                addStats(realOutput, 1,true); //
            } else {
                addStats(realOutput, 0, true); //
                addStats(scaledOutput, 1, false); //
            }
            count++;
        }

        String efficiency = String.format("%.3f",  (correct / (double) testData.size()));
        writeTestLine("test efficiency: " + efficiency);
    }

    private void addStats(double[] output, int value, boolean correct) {
        double[] setosa = new double[]{1.0, 0.0, 0.0};
        double[] versicolor = new double[]{0.0, 1.0, 0.0};
        double[] virginica = new double[]{0.0, 0.0, 1.0};
        int tf;
        if (correct) {
            tf = 1;
        } else {
            tf = 2;
        }

        if (Arrays.equals(output, setosa)) {
            outputStats[0][0]++;
            outputStats[0][tf] += value;
        } else if (Arrays.equals(output, versicolor)) {
            outputStats[1][0]++;
            outputStats[1][tf] += value;
        } else if (Arrays.equals(output, virginica)) {
            outputStats[2][0]++;
            outputStats[2][tf] += value;
        }
    }

    private double countError(Map.Entry<double[], double[]> current) {
        double[][][] gradient = countGradientDescent(current.getValue());
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
                writeTestLine("last layer errors: " + sb);
            }
        }
        return error;
    }

    /*--------------------------------------Math operations--------------------------------------*/

    private double sigmoid(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }

    private double sigmoidDerivative(double v) {
        return sigmoid(v) * (1 - sigmoid(v));
    }

    /*--------------------------------------String operations--------------------------------------*/

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

    /*--------------------------------------Statistics--------------------------------------*/

    double[] getScaledOutput() { // scale
        int outputIndex = activations.length - 1;
        double[] realOutput = activations[outputIndex];
        int outputSize = activations[outputIndex].length;

        int outputMax = 0;
        for (int i = 1; i < outputSize; i++) {
            if (realOutput[i] > realOutput[outputMax]) {
                outputMax = i;
            }
        }
        double[] scaledOutput = new double[outputSize];
        scaledOutput[outputMax] = 1;

        return scaledOutput;
    }

    public double[] getErrors() {
        return errors;
    }

    private void statsForTest(Map.Entry<double[], double[]> test, int count) {
        writeTestLine("TEST no. " + count + ":");
        writeTestLine("input: " + arrayToString(test.getKey()));

        writeTestLine("Global error: " + String.format("%.3f",  countError(test)));
        writeTestLine("Expected output: " + arrayToString(test.getValue()));

        // output layer
        writeTestLine("Output layer activations: ");
        int lastLayer = weights.length;
        writeTestLine(arrayToString(activations[lastLayer]));
        writeTestLine("Last layer weights & biases: ");
        for (int j = 0; j < weights[lastLayer - 1].length; j++) {
            writeTestLine(arrayToString(weights[lastLayer - 1][j]));
        }

        writeTestLine("-------------------------------");

        //hidden layers
        for (int i = weights.length - 2; i >= 0; i--) {
            if (sizesStats[i + 1]) {
                writeTestLine("Hidden layer activations");
                writeTestLine(arrayToString(activations[i + 1]));
                writeTestLine("Hidden layer weights & biases: ");
                for (int j = 0; j < weights[i].length; j++) {
                    writeTestLine(arrayToString(weights[i][j]));
                }
                writeTestLine("-------------------------------");
            }
        }

        writeTestLine("=======================================");
    }

    public int[][] getOutputStats() {
        return outputStats;
    }

    private void writeTrainLine(String line) {
        trainStats += line + "\n";
    }

    private void writeTestLine(String line) {
        testStats += line + "\n";
    }

    public String getTrainStats() {
        return trainStats;
    }

    public String getTestStats() {
        return testStats;
    }

}

