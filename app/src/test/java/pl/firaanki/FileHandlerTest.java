package pl.firaanki;

import junit.framework.TestCase;
import java.util.*;

public class FileHandlerTest extends TestCase {

    public void testRead() {

        Map<double[], double[]> data = FileHandler.getFile("iris.data").read();


    }
}