package pl.firaanki;

import junit.framework.TestCase;

import java.util.Map;
import java.util.Set;

public class ArraysTest extends TestCase {

    public void testGetWeights() {
        double[][][] weights = Arrays.getWeights(new int[]{4,7,3}, -1, 1);
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

    public void testGetBiases() {
        double[][] biases = Arrays.getBias(new int[]{4,7,3}, -1, 1);
        for (double[] db : biases) {
            System.out.println("---layer---");
            for (double dbs : db) {
                System.out.print(dbs + " ");
            }
            System.out.println();
        }
    }

    public void testGetWages() {
        double[] wages = Arrays.getWages(5, -1,1);
        for (double wage : wages) {
            System.out.print(wage + " ");
        }
    }

    public void testGetLayer() {
        double[][] layer = Arrays.getLayer(6, 3, -1, 1);
        for (double[] doubles : layer) {
            for (double aDouble : doubles) {
                System.out.print(aDouble + " ");
            }
            System.out.println();
        }
    }

    public void testGetNetwork() {
        double[][][] network = Arrays.getNetwork(new int[]{4,7,3}, -1 ,1);
        for (double[][] layer : network) {
            System.out.println("----layer-----");
            for (double[] doubles : layer) {
                for (double aDouble : doubles) {
                    System.out.print(aDouble + " ");
                }
                System.out.println();
            }
        }
    }
}