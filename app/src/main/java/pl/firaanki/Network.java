package pl.firaanki;

import java.util.*;

public class Network {

    double[][][] net;

    Network(int[] sizes, double min, double max) {
        net = Arrays.getNetwork(sizes, min, max);
    }



}
