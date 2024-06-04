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
    double[][] biases;
    double[][] activations;

    double[][][] velocityWeights;
    double[][] velocityBiases;
    transient double[][] sums;

    static final int MIN = -1;
    static final int MAX = 1;

    private transient String trainStats = "";
    private transient String testStats = "";
    private final int[][] stats = new int[3][2];

    boolean takeBias = false;
    boolean doShuffle = true;
    boolean[] sizesStats;

    Network(int[] sizes) {
        this.sizes = sizes;
        sizesStats = new boolean[sizes.length];
        writeTrainLine("Layers structure:");
        writeTrainLine(arrayToString(sizes));

        weights = ArrayLib.getWeights(sizes, MIN, MAX);
        biases = ArrayLib.getBias(sizes, MIN, MAX);
        activations = new double[sizes.length][];

        velocityWeights = new double[sizes.length - 1][][];
        velocityBiases = new double[sizes.length - 1][];
        for (int i = 0; i < sizes.length - 1; i++) {
            velocityWeights[i] = new double[sizes[i + 1]][sizes[i]];
            velocityBiases[i] = new double[sizes[i + 1]];
        }
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

    // todo: check error and global error
    // todo: run method


    public void onlineEpoch(List<Map.Entry<double[], double[]>> trainData, int epochs, double learningRate, double momentum) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double error = 0;
            error = doEpoch(trainData, learningRate, momentum, error);

            writeTrainLine("Training epoch no." + (epoch + 1));
            writeTrainLine("|-> error: " + String.format("%.3f", error));
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
                    biases[i][j] -= velocityBiases[i][j];
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
                // counts activation value for neuron
                activations[i][j] = sigmoid(sum) + biases[i - 1][j];
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
        for (Map.Entry<double[], double[]> test : testData) {
            countActivations(test.getKey());
            statsForTest(test, count);

            double[] scaledOutput = getScaledOutput();
            double[] realOutput = test.getValue();

            if (Arrays.equals(scaledOutput, realOutput)) {
                correct++;
                addStats(realOutput, 1,0); //
            } else {
                addStats(realOutput, 0, 0); //
                addStats(scaledOutput, 0, 1); //
            }
            count++;
        }

        String efficiency = String.format("%.3f",  (correct / (double) testData.size()));
        writeTestLine("test efficiency: " + efficiency);
    }
    // trueV = 1 -> true positive
    // falseV = 1 -> false positive
    // trueV = 0 -> true negative
    // falseV = 0 -> false negative
    private void addStats(double[] value, int trueV, int falseV) {
        double[] setosa = new double[]{1.0, 0.0, 0.0};
        double[] versicolor = new double[]{0.0, 1.0, 0.0};
        double[] virginica = new double[]{0.0, 0.0, 1.0};

        if (Arrays.equals(value, setosa)) {
            stats[0][0] += trueV; // true positive
            stats[0][1]++; // all presence
            stats[0][2] += falseV; // false positive
        } else if (Arrays.equals(value, versicolor)) {
            stats[1][0] += trueV;
            stats[1][1]++;
        } else if (Arrays.equals(value, virginica)) {
            stats[2][0] += trueV;
            stats[2][1]++;
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
            if (realOutput[i] > realOutput[i - 1]) {
                outputMax = i;
            }
        }
        double[] scaledOutput = new double[outputSize];
        scaledOutput[outputMax] = 1;

        return scaledOutput;
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
            writeTest(arrayToString(weights[lastLayer - 1][j]) + "| ");
            writeTestLine(String.valueOf(biases[lastLayer - 1][j]));
        }

        writeTestLine("-------------------------------");

        //hidden layers
        for (int i = weights.length - 2; i >= 0; i--) {
            if (sizesStats[i + 1]) {
                writeTestLine("Hidden layer activations");
                writeTestLine(arrayToString(activations[i + 1]));
                writeTestLine("Hidden layer weights & biases: ");
                for (int j = 0; j < weights[i].length; j++) {
                    writeTest(arrayToString(weights[i][j]) + "| ");
                    writeTestLine(String.valueOf(biases[i][j]));
                }
                writeTestLine("-------------------------------");
            }
        }

        writeTestLine("=======================================");
    }

    public int[][] getStats() {
        return stats;
    }

    private void writeTrainLine(String line) {
        trainStats += line + "\n";
    }

    private void writeTestLine(String line) {
        testStats += line + "\n";
    }

    private void writeTest(String line) {
        testStats += line;
    }

    public String getTrainStats() {
        return trainStats;
    }

    public String getTestStats() {
        return testStats;
    }

    /*-------------------------------Offline learning------------------------------------*/

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
            countActivations(current.getKey());
            gradient = countGradientDescent(current.getValue());

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
}

