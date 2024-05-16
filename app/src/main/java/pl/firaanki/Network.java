package pl.firaanki;

public class Network {

    double[][][] net;
    double[][] testExample;
    double[] gradientDescent;


    Network(int[] sizes, double min, double max) {
        net = Arrays.getNetwork(sizes, min, max);
    }

    public void runTest(double[][] testExample) {
        this.testExample = testExample;
    }
}
