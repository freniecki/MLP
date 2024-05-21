package pl.firaanki;

import junit.framework.TestCase;
import java.util.*;
import java.util.Arrays;

public class FileHandlerTest extends TestCase {

    public void testRead() {

        Map<double[], double[]> data = FileHandler.getFile("iris.data").read();

        for (Map.Entry<double[], double[]> element : data.entrySet()) {
            System.out.print(Arrays.toString(element.getKey()));
            System.out.print(" | ");
            System.out.println(Arrays.toString(element.getValue()));
        }
    }
}