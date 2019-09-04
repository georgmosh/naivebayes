import java.util.ArrayList;
import java.util.HashSet;

/**
 *  TestLogic.java
 *  This class illustrates a sample complete Naive Bayes Test; including feature selection.
 *   @author Georgios M. Moschovis (p3150113@aueb.gr)
 */
public class TestLogic {
    private NaiveBayes thisvoc;
    private GraphicsDAO simpleNB, k_best_FS;
    private ArrayList<String> train_data_filenames;
    private HashSet<String> k_best_properties;

    private String trainDataRepository, testDataRepository, dataset;
    private double kappa;

    /**
     *  Default comparator override.
     * @param trainRep The train data repository.
     * @param testRep The test data repository.
     * @param dataset The Naive Bayes dataset for the statistics.
     * @param k The data percentage to be used from the vocabulary dataset; as the k-best.
     */
    public TestLogic(String trainRep, String testRep, String dataset, double k) {
        thisvoc = new NaiveBayes();
        this.trainDataRepository = trainRep;
        this.testDataRepository = testRep;
        this.dataset = dataset;
        this.kappa = k;
        this.simpleNB = new GraphicsDAO("Simple Naive Bayes: Statistics - " + dataset, "Simple");
        this.k_best_FS = new GraphicsDAO("k-Best Feature Selection NB: Statistics - " + dataset, "k-Best Feature Selection");
    }


    /**
     *  A method enabling calculation of the a priori probabilities, as well as a priori enthropy, for the appearences of each
     *  category; using the precalculated a priori probabilities but not the context of the datasets. Later on, also permits computations
     *  of a posteriori probabilities for the appearence of each word in the vocabulary dataset used for training. For further information
     *  also check documentation for the GetAPrioriProbabilities(), GetAPrioriEntrophy(), GetAPosterioriProbabilities() methods
     *  or the NaiveBayes class.
     */
    public void TrainNaiveBayes() {
        train_data_filenames = thisvoc.listAllFiles(".\\src\\" + trainDataRepository);
        thisvoc.numOfTrainData = train_data_filenames.size();
        for(int train = 0; train < thisvoc.numOfTrainData; train++) {
            thisvoc.ReadTrainFile("src/" + trainDataRepository + "/" + train_data_filenames.get(train));
        }
        thisvoc.GetAPrioriProbabilities();
        thisvoc.GetAPrioriEntrophy();
        thisvoc.GetAPosterioriProbabilities();
    }

    /**
     *  A naive method calculating estimation probabilities for the category of a train document; given the a priori, as well as a
     *   posteriori probabilities for the appearence of each word in the vocabulary dataset; including k-Best features usage for the
     *   computations, selected based on their pre-calculated information gain, picturing how uncertain we are about determining a
     *   document's category. Thus, we are interested in minimizing the a posteriori entrophy. For further information also check the
     *   documentation of GetInformationGainConclusions(), GetClassificationConclusions(), GetKBestClassificationConclusions()
     *   methods of the NaiveBayes class. Finally, connects the Naive Bayes implementation storage with the graphical output,
     *   enabling setting value for NB performances on the train dataset graphics DAO.
     */
    public void TestOnTrainData() {
        k_best_properties = thisvoc.GetInformationGainConclusions(kappa);
        ArrayList<Vec2<Integer,Double>> graphData = null, k_best_graphData = null;
        for(int train = 0; train< thisvoc.numOfTrainData; train++) {
            Vec2<String, ArrayList<String>> testData = thisvoc.ReadTestFile("src/" + trainDataRepository + "/" + train_data_filenames.get(train));
            thisvoc.GetClassificationConclusions(testData, 'T');
            thisvoc.GetKBestClassificationConclusions(testData, k_best_properties, 'T');
        }
        simpleNB.LoadTrainData(thisvoc.dataPlotPts);
        k_best_FS.LoadTrainData(thisvoc.k_best_dataPlotPts);
        OutputResults();
    }

    /**
     *  A naive method calculating estimation probabilities for the category of a test document; given the a priori, as well as a
     *   posteriori probabilities for the appearence of each word in the vocabulary dataset; including k-Best features usage for the
     *   computations, selected based on their pre-calculated information gain, picturing how uncertain we are about determining a
     *   document's category. Thus, we are interested in minimizing the a posteriori entrophy. For further information also check the
     *   documentation of GetInformationGainConclusions(), GetClassificationConclusions(), GetKBestClassificationConclusions()
     *   methods of the NaiveBayes class. Finally, connects the Naive Bayes implementation storage with the graphical output,
     *   enabling setting value for NB performances on the test dataset graphics DAO.
     */
    public void TestOnRealData() {
        thisvoc.reset();
        ArrayList<String> test_data_filenames = thisvoc.listAllFiles(".\\src\\" + testDataRepository);
        thisvoc.numOfTestData = test_data_filenames.size();
        for(int test = 0; test< thisvoc.numOfTestData; test++) {
            Vec2<String, ArrayList<String>> testData = thisvoc.ReadTestFile("src/" + testDataRepository + "/" + test_data_filenames.get(test));
            thisvoc.GetClassificationConclusions(testData, 'R');
            thisvoc.GetKBestClassificationConclusions(testData, k_best_properties, 'R');
        }
        simpleNB.LoadRealData(thisvoc.dataPlotPts);
        k_best_FS.LoadRealData(thisvoc.k_best_dataPlotPts);
        OutputResults();
        simpleNB.Run();
        k_best_FS.Run();
    }

    /**
     *  Method outputting Naive Bayes results to default output stream (System.out).
     */
    public void OutputResults() {
        int right = thisvoc.TP + thisvoc.TN, wrong = thisvoc.FP + thisvoc.FN;
        double precision = thisvoc.TP/(double)(thisvoc.TP + thisvoc.FP);
        double recall = thisvoc.TP/(double)(thisvoc.TP + thisvoc.FN);
        double F1 = 2 * precision * recall / (precision + recall);
        System.out.println("Right predictions: " + right + ", " + Math.round(((double)right / (right + wrong)) * 100) + "% and wrong predictions: " + wrong);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
        System.out.println("F1: " + F1);
        System.out.println("\n*** USING FEATURE SELECTION ***");
        right = thisvoc.k_best_TP + thisvoc.k_best_TN; wrong = thisvoc.k_best_FP + thisvoc.k_best_FN;
        precision = thisvoc.k_best_TP/(double)(thisvoc.k_best_TP + thisvoc.k_best_FP);
        recall = thisvoc.k_best_TP/(double)(thisvoc.k_best_TP + thisvoc.k_best_FN);
        F1 = 2 * precision * recall / (precision + recall);
        System.out.println("Right k-best predictions: " + right + ", " + Math.round(((double)right / (right + wrong)) * 100) + "% and wrong predictions: " + wrong);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " +  recall);
        System.out.println("F1: " + F1 + "\n");
    }
}