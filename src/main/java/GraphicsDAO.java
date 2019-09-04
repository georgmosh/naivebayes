import org.jfree.ui.RefineryUtilities;

import java.util.ArrayList;

/**
 *  GraphicsDAO.java
 *  This class illustrates a sample data base object implementation for Graphics.
 *   @author Georgios M. Moschovis (p3150113@aueb.gr)
 */
public class GraphicsDAO {
    /*
     *  Local variables.
     */
    private String graphWindowTitle, naiveBayesType, dataset;
    private ArrayList<Vec2<Integer, Double>> sampleTrainData, sampleRealData;

    /**
     *  Default constructor override.
     *  @param title the Graphics window frame title.
     *  @param type the Naive Bayes type for the statistics.
     */
    public GraphicsDAO(String title, String type) {
        this.graphWindowTitle = title; this.naiveBayesType = type;
    }

    /**
     *  Method setting value for NB performances on the train dataset.
     *  @param sampleTrainData The train data statistics.
     */
    protected void LoadTrainData(ArrayList<Vec2<Integer, Double>> sampleTrainData) {
        this.sampleTrainData = sampleTrainData;
    }

    /**
     *  Method setting value for NB performances on the test dataset.
     *  @param sampleRealData The test data statistics.
     */
    protected void LoadRealData(ArrayList<Vec2<Integer, Double>> sampleRealData) {
        this.sampleRealData = sampleRealData;
    }

    /**
     *  Method demonstrating Graphics Application.
     */
    public void Run() {
        final Graphics demo = new Graphics(graphWindowTitle, naiveBayesType, sampleTrainData, sampleRealData);
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }
}
