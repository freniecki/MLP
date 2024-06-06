package pl.firaanki;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import javax.swing.JFrame;

public class Plot extends JFrame {
    public Plot(String title, double[] errors) {
        super(title);

        DefaultCategoryDataset dataset = createDataset(errors);

        JFreeChart chart = ChartFactory.createLineChart(
                "Error Plot",
                "Epochs",
                "Error Value",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        // Create Panel
        ChartPanel panel = new ChartPanel(chart);
        setContentPane(panel);
    }

    private DefaultCategoryDataset createDataset(double[] errors) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (int i = 0; i < errors.length; i++) {
            dataset.addValue(errors[i], "Errors", Integer.toString(i + 1));
        }

        return dataset;
    }
}
