
import java.util.Comparator;

/**
 * DefaultComparator.java
 * @author Georgios M. Moschovis (p3150113@aueb.gr)
 */
final class DefaultComparator implements Comparator {
    public int compare(Object a, Object b) {
        return ((Comparable)a).compareTo(b);
    }
}