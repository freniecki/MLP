package pl.firaanki;

public class Statistics {
    private final int[][] stats = new int[4][2];

    Statistics(int[][] stats) {
        for (int i = 0; i < 3; i++) {
            System.arraycopy(stats[i], 0, this.stats[i], 0, 2);
        }
        stats[3][0] = stats[0][0] + stats[1][0] + stats[2][0];
        stats[3][1] = stats[0][1] + stats[1][1] + stats[2][1];
    }

    String getClassification() {
        StringBuilder sb = new StringBuilder();
        sb.append("Setosa: ").append(stats[0][0]).append("/").append(stats[0][1]).append("\n");
        sb.append("Versicolor: ").append(stats[1][0]).append("/").append(stats[1][1]).append("\n");
        sb.append("Virginica: ").append(stats[2][0]).append("/").append(stats[2][1]).append("\n");
        sb.append("Ogółem: ").append(stats[3][0]).append("/").append(stats[3][1]).append("\n");
        return sb.toString();
    }

    String getPrecision() {
        return "";
    }

    String getRecall() {
        return "";
    }

    String getFmeasure() {
        return "";
    }

    String getAllStats() {
        String separator = "-----------------";
        return getClassification() + "\n" +
                separator +
                getPrecision() + "\n" +
                separator +
                getRecall() + "\n" +
                separator +
                getFmeasure() + "\n" +
                separator;
    }
}
