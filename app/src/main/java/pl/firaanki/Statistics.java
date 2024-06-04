package pl.firaanki;

public class Statistics {
    private final int[][] stats = new int[4][2];
    int TP;
    int FP;
    int FN;

    Statistics(int[][] stats) {
        for (int i = 0; i < 3; i++) {
            System.arraycopy(stats[i], 0, this.stats[i], 0, 2);
        }
        this.stats[3][0] = stats[0][0] + stats[1][0] + stats[2][0];
        this.stats[3][1] = stats[0][1] + stats[1][1] + stats[2][1];

        TP = stats[0][1] + stats[1][1] + stats[2][1];
        FP = stats[0][2] + stats[1][2] + stats[2][2];
        FN = 0;
    }

    String getClassification() {
        StringBuilder sb = new StringBuilder();
        sb.append("Setosa: ").append(stats[0][1]).append("/").append(stats[0][0]).append("\n");
        sb.append("Versicolor: ").append(stats[1][1]).append("/").append(stats[1][0]).append("\n");
        sb.append("Virginica: ").append(stats[2][1]).append("/").append(stats[2][0]).append("\n");
        sb.append("Ogółem: ").append(stats[3][1]).append("/").append(stats[3][0]).append("\n");
        return sb.toString();
    }

    String getErrorMatrix() {
        int TN = stats[3][0] - stats[3][1];
        return TP + " | " + FP + "\n" + FN + " | " + TN;
    }

    double getPrecision() {
        return (double) TP / (TP + FP);
    }

    double getRecall() {
        return (double) TP / (TP + FN);
    }

    double getFmeasure() {
        double mianownik = (1 / getRecall()) + (1 / getPrecision());
        return 1 / mianownik;
    }

    String getAllStats() {
        return getClassification() + "\n" +
                "Error matrix: " + "\n" +
                getErrorMatrix() + "\n" +
                "Precision: " + String.format("%.3f", getPrecision()) + "\n" +
                "Recall: " + String.format("%.3f", getRecall()) + "\n" +
                "F-measure: " + String.format("%.3f", getFmeasure()) + "\n";
    }
}
