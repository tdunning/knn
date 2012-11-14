package org.apache.mahout.knn.remove;

import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;

@RunWith(value = Parameterized.class)
public class SyntheticDataUtilsTest {
  private int numDimensions;
  private int numDatapoints;
  private double distributionRadius;

  public SyntheticDataUtilsTest(int numDimensions, int numDatapoints, double distributionRadius) {
    this.numDimensions = numDimensions;
    this.numDatapoints = numDatapoints;
    this.distributionRadius = distributionRadius;
  }

  /**
   * Generates data for parameterized tests.
   * The structure of the parameters is as follows:
   * numDimensions numDatapoints distributionRadius
   * These are used as arguments when calling DataUtils.sampleMultiNormalHypercube.
   *
   * @return a collection of bindings for each parameter.
   * @see org.apache.mahout.knn.cluster.DataUtils#sampleMultiNormalHypercube(int, int, double)
   */
  @Parameters
  public static Collection<Object[]> generateData() {
    return Arrays.asList(new Object[][] {
        {3, 20, 0.001},
        {8, 400, 0.1},
        {10, 20000, 0.4},
    });
  }

  @Test
  public void testSampleMultiNormalHypercubeDistribution() {
    Pair<List<Centroid>, List<Centroid>> data =
        DataUtils.sampleMultiNormalHypercube(numDimensions, numDatapoints,
            distributionRadius);
    BruteSearch searcher = new BruteSearch(new EuclideanDistanceMeasure());
    for (Vector mean : data.getSecond()) {
      searcher.add(mean);
    }
    for (Centroid point : data.getFirst()) {
      WeightedThing<Vector> closest = searcher.search(point, 1).get(0);
      assertThat(closest.getWeight(), lessThan(distributionRadius));
    }
  }
}
