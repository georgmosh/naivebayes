import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.util.ArrayList;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

/**
 * Graphics.java
 * A simple demonstration application showing how to create a line chart using data from a
 * {@link CategoryDataset}.
 * @author Georgios M. Moschovis (p3150113@aueb.gr)
 */
public class Graphics extends ApplicationFrame {
    /*
     *  Local variables.
     */
    private String naiveBayesType;
    private ArrayList<Vec2<Integer, Double>> sampleTrainData, sampleRealData;

    /**
     * Creates a new demo.
     * Default constructor override.
     * @param title the Graphics window frame title.
     * @param type the Naive Bayes type for the statistics.
     * @param sampleTrainData Train progress datapoints.
     * @param sampleRealData Test progress datapoints.
     */
    public Graphics(final String title, String type, ArrayList<Vec2<Integer, Double>> sampleTrainData, ArrayList<Vec2<Integer, Double>> sampleRealData) {
        super(title);
        this.naiveBayesType = type;
        this.sampleTrainData = sampleTrainData;
        this.sampleRealData = sampleRealData;
        final CategoryDataset dataset = createDataset();
        final JFreeChart chart = createChart(dataset);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(500, 270));
        setContentPane(chartPanel);
    }
    /**
     * Creates a sample dataset.
     *
     * @return The dataset.
     */
    private CategoryDataset createDataset() {
        // row keys...
        final String series1 = "Train Data Accuracy";
        final String series2 = "Test Data Accuracy";

        // create the dataset...
        final DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for(Vec2<Integer, Double> data: sampleTrainData) {
            dataset.addValue(data.getYValue(), series1, data.getTValue());
        }
        for(Vec2<Integer, Double> data: sampleRealData) {
            dataset.addValue(data.getYValue(), series2, data.getTValue());
        }

        return dataset;
    }

    /**
     * Creates a sample chart.
     *
     * @param dataset a dataset.
     * @return The chart.
     */
    private JFreeChart createChart(final CategoryDataset dataset) {
        // create the chart...
        final JFreeChart chart = ChartFactory.createLineChart(
                this.naiveBayesType + " Naive Bayes Performance",       // chart title
                "Percentage of Data Used (%)",                    // domain axis label
                "Accuracy Value",                   // range axis label
                dataset,                   // data
                PlotOrientation.VERTICAL,  // orientation
                true,                      // include legend
                true,                      // tooltips
                false                      // urls
        );

        // NOW DO SOME OPTIONAL CUSTOMISATION OF THE CHART...
//        final StandardLegend legend = (StandardLegend) chart.getLegend();
        //      legend.setDisplaySeriesShapes(true);
        //    legend.setShapeScaleX(1.5);
        //  legend.setShapeScaleY(1.5);
        //legend.setDisplaySeriesLines(true);

        chart.setBackgroundPaint(Color.white);

        final CategoryPlot plot = (CategoryPlot) chart.getPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setRangeGridlinePaint(Color.white);

        // customise the range axis...
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        rangeAxis.setAutoRangeIncludesZero(true);

        // ****************************************************************************
        // * JFREECHART DEVELOPER GUIDE                                               *
        // * The JFreeChart Developer Guide, written by David Gilbert, is available   *
        // * to purchase from Object Refinery Limited:                                *
        // *                                                                          *
        // * http://www.object-refinery.com/jfreechart/guide.html                     *
        // *                                                                          *
        // * Sales are used to provide funding for the JFreeChart project - please    *
        // * support us so that we can continue developing free software.             *
        // ****************************************************************************

        // customise the renderer...
        final LineAndShapeRenderer renderer = (LineAndShapeRenderer) plot.getRenderer();
//        renderer.setDrawShapes(true);

        renderer.setSeriesStroke(
                0, new BasicStroke(
                        2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                        1.0f, new float[]{10.0f, 6.0f}, 0.0f
                )
        );
        renderer.setSeriesStroke(
                1, new BasicStroke(
                        2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                        1.0f, new float[]{6.0f, 6.0f}, 0.0f
                )
        );
        renderer.setSeriesStroke(
                2, new BasicStroke(
                        2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                        1.0f, new float[]{2.0f, 6.0f}, 0.0f
                )
        );
        // OPTIONAL CUSTOMISATION COMPLETED.

        return chart;
    }
}

