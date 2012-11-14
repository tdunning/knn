package org.apache.mahout.knn.remove;

import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.math.Centroid;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.List;

import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertThat;

@RunWith(value = Parameterized.class)
public class DataUtilsTest {
  private int numDimensions;
  private int numDatapoints;
  private double distributionRadius;

  public DataUtilsTest(int numDimensions, int numDatapoints, double distributionRadius) {
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
  @Parameterized.Parameters
  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][]{
        {3, 20, 0.001},
        {8, 400, 0.001},
        {10, 20000, 0.004},
    });
  }
  @Test
  public void testEstimateDistanceCutoff() {
    List<Centroid> data =
        DataUtils.sampleMultiNormalHypercube(numDimensions, numDatapoints, distributionRadius).getFirst();
    assertThat("Impossible minimum distance in hypercube",
        DataUtils.estimateDistanceCutoff(data), is(lessThan(0.5)));
  }
}
