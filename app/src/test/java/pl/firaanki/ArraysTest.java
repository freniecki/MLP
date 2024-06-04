package pl.firaanki;

import junit.framework.TestCase;

public class ArraysTest extends TestCase {

    public void testGetWeights() {
        double[][][] weights = ArrayLib.getWeights(new int[]{4,7,3}, -1, 1);
        for (double[][] db : weights) {
            System.out.println("---layer---");
            for (double[] dbs : db) {
                for (double dbsm : dbs) {
                    System.out.print(dbsm + " ");
                }
                System.out.println();
            }
        }
    }
}