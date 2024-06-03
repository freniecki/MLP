package pl.firaanki;

public class ArrayLib {

    private ArrayLib() {
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
            bias[i] = new double[sizes[i + 1]];
            for (int j = 0; j < bias[i].length; j++) {
                // set for 0's
                bias[i][j] = 0.0;
                // set random
                //bias[i][j] = min + Math.random() * (max - min);
            }
        }

        return bias;
    }

}
