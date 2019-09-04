/**
 *  Tests.java
 *  Executable file proceeding with all Naive Bayes designed tests.
 *  @author Georgios M. Moschovis (p3150113@aueb.gr)
 */
public class Tests {
    /**
     *  Test on Enron-Spam dataset.
     */
    public static void EnronSpam() {
        TestLogic current = new TestLogic("enron_train_data", "enron_test_data", "Enron-Spam", 0.9);
        current.TrainNaiveBayes();
        current.TestOnTrainData();
        current.TestOnRealData();
    }

    /**
     *  Main Method.
     */
    public static void main(String[] args) {
        System.out.println("Example: Enron-Spam"); EnronSpam();
    }
}
