import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.apache.commons.math3.analysis.function.Abs;
import org.apache.commons.math3.analysis.function.Log;

/**
 *  NaiveBayes.java
 *  @author Georgios M. Moschovis (p3150113@aueb.gr)
 */
public class NaiveBayes {
    /**
     * Static field used as data accessor between readFile & the vocabulary instances; implementing
     * a parameterized tokens dynamic container if reading was successful, being assigned contents of the
     * datasets, otherwise NULL.
     */
    protected static HashMap<String, Vec2<Vec2<Integer, Integer>, Vec2<Double,Double>>> vocabulary;

    /**
     *  The actual data classification results.
     */
    protected ArrayList<Vec2<String, Double>> results;

    /**
     *  The k-best properties selection data classification results.
     */
    protected ArrayList<Vec2<String, Double>> k_best_results;

    /**
     *  The documents classification statistics.
     */
    ArrayList<Vec2<Integer, Double>> dataPlotPts;

    /**
     *  The k-best properties selection documents classification statistics.
     */
    ArrayList<Vec2<Integer, Double>> k_best_dataPlotPts;

    /**
     *  The a posteriori entrophy values per category.
     */
    protected ArrayList<Vec2<String,Vec2<Double, Double>>> a_posteriori_entrophy;

    /*
    * Local variables.
    */
    protected static int currentClassifications = 0, k_best_currentClassifications = 0;
    protected int TP, TN, FP, FN, k_best_TP, k_best_TN, k_best_FP, k_best_FN;
    protected int spam_counter, appearences, numOfTestData, numOfTrainData;
    protected double spam_probability, ham_probability, a_priori_entrophy;

    /**
     * Default comparator override.
     */
    public NaiveBayes() {
        vocabulary = new HashMap<String, Vec2<Vec2<Integer, Integer>, Vec2<Double,Double>>>();
        spam_counter = appearences = 0;
    }

    /**
     * A method computing base-2 logarithms.
     * @param x The decimal number to calculate base-2 logarithm.
     * @return The base-2 logarithm computed.
     */
    public double log2(double x) {
        Log log = new Log();
        return log.value(x) / log.value(2);
    }

    /**
     *  A method classifying a train message as spam.
     * @param filename The filename of the train message.
     * @return true if the message is spam.
     */
     public boolean isSpam(String filename) {
         return (filename.contains("spam") || filename.contains("spm"));
     }

    /**
     * A method reading a given train file of a specific format described in comments; contents of which are
     * assigned to a parameterized tokens dynamic container if reading was successful; otherwise displays an
     * error description.
     * @param filename The name of the given text file.
     */
    public void ReadTrainFile(String filename) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(filename));
            if (isSpam(filename)) spam_counter++;
            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null) {
                sCurrentLine = sCurrentLine.trim().replaceAll(" +", " "); // unify whitespaces
                String[] tuples_context = sCurrentLine.split(" "); // separator is: " " = [whitespace]
                for(String token: tuples_context) {
                    if (token.length() > 0 && (!Character.isLetter(token.charAt(token.length() - 1)))) token = token.substring(0, token.length() - 1);
                    if (token.length() > 1) {
                        if (vocabulary.containsKey(token)) {
                            int current = vocabulary.get(token).getTValue().getTValue();
                            vocabulary.get(token).getTValue().setTValue(current + 1);
                            appearences++;
                            if (isSpam(filename)) {
                                current = vocabulary.get(token).getTValue().getYValue();
                                vocabulary.get(token).getTValue().setYValue(current + 1);
                            }
                        } else {
                            int isSpam = 0;
                            if (isSpam(filename)) isSpam++;
                            Vec2<Integer, Integer> probs = new Vec2<Integer, Integer>(1, isSpam);
                            Vec2<Vec2<Integer, Integer>, Vec2<Double, Double>> wordVec = new Vec2<Vec2<Integer, Integer>, Vec2<Double, Double>>(probs, new Vec2<Double, Double>());
                            vocabulary.put(token, wordVec);
                        }
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("The system could not find the file specified");
            e.printStackTrace();
        } finally { // Stream closure should be executed at any case
            try {
                if (br != null) br.close();
            } catch (IOException exception) {
                exception.printStackTrace();
            }
        }
    }

    /**
     * A method reading a given test file of a specific format described in comments; contents of which
     * assigned to a parameterized tokens dynamic container if reading was successful; otherwise displays an
     * error description.
     * @param filename The name of the given text file.
     * @return The vocabulary inclusive tokens dynamic container.
     */
    public Vec2<String, ArrayList<String>> ReadTestFile(String filename) {
        BufferedReader br = null;
        ArrayList<String> containingTokens = new ArrayList<String>();
        try {
            br = new BufferedReader(new FileReader(filename));
            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null) {
                sCurrentLine = sCurrentLine.trim().replaceAll(" +", " "); // unify whitespaces
                String[] tuples_context = sCurrentLine.split(" "); // separator is: " " = [whitespace]
                for(String token: tuples_context) {
                    if (token.length() > 0 && (token.charAt(token.length() - 1)) == ('.')) token = token.substring(0, token.length() - 1);
                    if (vocabulary.containsKey(token)) containingTokens.add(token);
                }
            }
        } catch (IOException e) {
            System.err.println("The system could not find the file specified");
            e.printStackTrace();
        } finally { // Stream closure should be executed at any case
            try {
                if (br != null) br.close();
            } catch (IOException exception) {
                exception.printStackTrace();
            }
            return new Vec2<String, ArrayList<String>>(filename, containingTokens);
        }
    }

    /**
     *  A method calculating a priori probabilities for the appearences of each category; without using the context
     *  of the datasets. This information is stored in the existent probability local variables to be used later on for
     *  computing classification estimates.
     */
    public void GetAPrioriProbabilities() {
        spam_probability = spam_counter/(double)numOfTrainData;
        ham_probability = 1.0 - spam_probability;
    }

    /**
     *  A method calculating a priori entrophy for the appearences of both categories; using the precalculated a
     *  priori probabilities but not the context of the datasets. This information is stored in the existent entrophy local
     *  variables to be used later on for computing information gain and select best classification properties for
     *  estimates computations.
     */
    public void GetAPrioriEntrophy() {
        a_priori_entrophy = -(spam_probability * log2(spam_probability)) -(ham_probability * log2(ham_probability));
    }

    /**
     *  A method calculating a posteriori probabilities for the appearence of each word in the vocabulary dataset
     *  used for training, to be stored in the existent two-dimensional vectors and be later on used for computing "naive"
     *  classification estimates for the mail documents. The term "Naive" is adopted because of the naive independence
     *   assumption between every pair of features.
     */
    public void GetAPosterioriProbabilities() {
        Set<String> words = vocabulary.keySet();
        Iterator<String> currentToken = words.iterator();
        while(currentToken.hasNext()) {
            String nextToken = currentToken.next();
            int nodal = vocabulary.get(nextToken).getTValue().getTValue();
            int spam = vocabulary.get(nextToken).getTValue().getYValue();
            vocabulary.get(nextToken).getYValue().setTValue((spam + 1.0)/(nodal + 2.0));
            vocabulary.get(nextToken).getYValue().setYValue(((nodal - spam) + 1.0)/(nodal + 2.0));
        }
    }

    /**
     *  A naive method calculating estimation probabilities for the category of a document; given the a priori, as
     *  well as a posteriori probabilities for the appearence of each word in the vocabulary dataset used for training.
     *  The term "Naive" is being adopted because of the naive independence assumption between every pair of
     *  features. LaPlace estimations are included in the computations; to avoid zero values in probabilities due to
     *  one property.
     *  @param fileData The vocabulary inclusive tokens dynamic container.
     */
    public void GetClassificationConclusions(Vec2<String, ArrayList<String>> fileData, char dataType) {
        ArrayList<String> tokens = fileData.getYValue();
        if(results == null) results = new ArrayList<Vec2<String, Double>>();
        if(dataPlotPts == null) dataPlotPts = new ArrayList<Vec2<Integer, Double>>();
        Vec2<Double, Double> probabilityProducts = new Vec2<Double, Double>(1.0 /* Spam initial probability */, 1.0 /* Ham initial probability */);
        for(String word: tokens) {
            probabilityProducts.setTValue(probabilityProducts.getTValue() * vocabulary.get(word).getYValue().getTValue());
            probabilityProducts.setYValue(probabilityProducts.getYValue() * vocabulary.get(word).getYValue().getYValue());
        }
        probabilityProducts.setTValue(probabilityProducts.getTValue() * spam_probability);
        probabilityProducts.setYValue(probabilityProducts.getYValue() * ham_probability);
        if(probabilityProducts.getTValue() < probabilityProducts.getYValue()) {
            results.add(new Vec2<String, Double>(fileData.getTValue() + " HAM", probabilityProducts.getYValue()));
            if (!isSpam(fileData.getTValue())) TN++; /* True Negative */ else FN++; /* False Negative */
        } else {
            results.add(new Vec2<String, Double>(fileData.getTValue() + " SPAM", probabilityProducts.getTValue()));
            if(isSpam(fileData.getTValue())) TP++; /* True Positive */ else FP++; /* False Positive */
        }
        int current = currentClassifications;
        if (dataType == 'R') {
            if(current == ((dataPlotPts.size() + 1) * numOfTestData / 10) - 1 || current == numOfTestData - 1)
                dataPlotPts.add(new Vec2<Integer, Double>((dataPlotPts.size() + 1) * 10, (TN+TP)/(double)(TN+FN+TP+FP) /* Accuracy */));
        } else if (dataType == 'T') {
            if(current == ((dataPlotPts.size() + 1) * numOfTrainData / 10) - 1 || current == numOfTrainData - 1)
               dataPlotPts.add(new Vec2<Integer, Double>((dataPlotPts.size() + 1) * 10, (TN+TP)/(double)(TN+FN+TP+FP) /* Accuracy */));
        }
        currentClassifications++;
    }

    /**
     *  A method computing how uncertain we are about determining a document's category; or equivalently what is
     *  the minimum amount of information we are provided to certainly specify the document category; or equivalently,
     *  supposing an ideal information encoding, what is the expected necessary number of bits to be transmitted; in
     *  order to correctly estimate the document category. This a posteriori entrophy is used to compute the information
     *  gain for each word; as the expected decrease of this amount.
     *  @param k The data percentage to be used from the vocabulary dataset; as the k-best.
     *  @return The k-Best properties; based on their pre-calculated information gain.
     */
    public HashSet<String> GetInformationGainConclusions(double k) {
        Set<String> words = vocabulary.keySet();
        Iterator<String> currentToken = words.iterator();
        if(a_posteriori_entrophy == null) a_posteriori_entrophy = new ArrayList<Vec2<String,Vec2<Double, Double>>>();
        while(currentToken.hasNext()) {
            String word = currentToken.next();
            Vec2<Double, Double> entrophyInfoGain = new Vec2<Double, Double>();
            double spam_entrophy = spam_probability * vocabulary.get(word).getYValue().getTValue();
            double ham_entrophy = ham_probability * vocabulary.get(word).getYValue().getYValue();
            entrophyInfoGain.setTValue(-(spam_entrophy * log2(spam_entrophy) -(ham_entrophy * log2(ham_entrophy))));
            double word_nodal_probability = vocabulary.get(word).getTValue().getTValue();
            word_nodal_probability /= appearences;
            entrophyInfoGain.setYValue(a_priori_entrophy -(word_nodal_probability * spam_entrophy) -(word_nodal_probability * ham_entrophy));
            a_posteriori_entrophy.add(new Vec2<String,Vec2<Double, Double>>(word, entrophyInfoGain));
        }
        a_posteriori_entrophy.sort(new DefaultComparator());
        HashSet<String> k_best_properties = new HashSet<String>();
        Abs abs = new Abs();
        Double start = (abs.value(1.0 - k) * a_posteriori_entrophy.size());
        for(int i = start.intValue(); i < a_posteriori_entrophy.size(); i++) k_best_properties.add(a_posteriori_entrophy.get(i).getTValue());
        System.out.println("DIMENSIONALITY REDUCTION \nINITIAL: " + vocabulary.size() + " REMAINING: " + k_best_properties.size() + "\n");
        return k_best_properties;
    }

    /**
     *  A naive method calculating estimation probabilities for the category of a document; given the a priori, the
     *  a posteriori probabilities for the appearence of each word in the vocabulary dataset used for training; as well.
     *  as the a priori and a posteriori values of entrophy. The term "Naive" is being adopted because of the naive
     *  independence assumption between every pair of features. k-Best features are used for the computations,
     *  selected based on their pre-calculated information gain, and LaPlace estimations; to avoid zero values in
     *  probabilities due to one property.
     *  @param fileData The vocabulary inclusive tokens dynamic container.
     */
    public void GetKBestClassificationConclusions(Vec2<String, ArrayList<String>> fileData, HashSet<String> k_best_properties, char dataType) {
        ArrayList<String> tokens = fileData.getYValue();
        if(k_best_results == null) k_best_results = new ArrayList<Vec2<String, Double>>();
        if(k_best_dataPlotPts == null) k_best_dataPlotPts = new ArrayList<Vec2<Integer, Double>>();
        Vec2<Double, Double> probabilityProducts = new Vec2<Double, Double>(1.0 /* Spam initial probability */, 1.0 /* Ham initial probability */);
        for(String word: tokens) {
            if(k_best_properties.contains(word)) {
                probabilityProducts.setTValue(probabilityProducts.getTValue() * vocabulary.get(word).getYValue().getTValue());
                probabilityProducts.setYValue(probabilityProducts.getYValue() * vocabulary.get(word).getYValue().getYValue());
            }
        }
        probabilityProducts.setTValue(probabilityProducts.getTValue() * spam_probability);
        probabilityProducts.setYValue(probabilityProducts.getYValue() * ham_probability);
        if(probabilityProducts.getTValue() < probabilityProducts.getYValue()) {
            k_best_results.add(new Vec2<String, Double>(fileData.getTValue() + " HAM", probabilityProducts.getYValue()));
            if (!isSpam(fileData.getTValue())) k_best_TN++; /* True Negative */ else k_best_FN++; /* False Negative */
        } else {
            k_best_results.add(new Vec2<String, Double>(fileData.getTValue() + " SPAM", probabilityProducts.getTValue()));
            if(isSpam(fileData.getTValue())) k_best_TP++; /* True Positive */ else k_best_FP++; /* False Positive */
        }
        int current = k_best_currentClassifications;
        if (dataType == 'R') {
            if(current == ((k_best_dataPlotPts.size() + 1) * numOfTestData / 10) - 1 || current == numOfTestData - 1)
                k_best_dataPlotPts.add(new Vec2<Integer, Double>((k_best_dataPlotPts.size() + 1) * 10, (TN+TP)/(double)(TN+FN+TP+FP) /* Accuracy */));
        } else if (dataType == 'T') {
            if(current == ((k_best_dataPlotPts.size() + 1) * numOfTrainData / 10) - 1 || current == numOfTrainData - 1)
                k_best_dataPlotPts.add(new Vec2<Integer, Double>((k_best_dataPlotPts.size() + 1) * 10, (TN+TP)/(double)(TN+FN+TP+FP) /* Accuracy */));
        }
        k_best_currentClassifications++;
    }

    /**
     * A method creating a dynamic container of all filepaths contained in a given directory.
     * @param path The directory to find all contained filepaths.
     */
    public ArrayList<String> listAllFiles(String path){
        ArrayList<String> filenames = new ArrayList<String>();
        File folder = new File(path);
        File[] listOfFiles = folder.listFiles();
        System.out.println(folder);
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) filenames.add(listOfFiles[i].getName());
        }
        return filenames;
    }

    /**
     *  A method resetting methods' data exchange points.
     */
    protected void reset() {
        results = k_best_results = null;
        TP = TN = FP = FN = k_best_TP = k_best_TN = k_best_FP = k_best_FN = 0;
        currentClassifications = k_best_currentClassifications = 0;
        dataPlotPts = k_best_dataPlotPts = null;
    }
}
