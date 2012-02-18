package org.apache.mahout.knn.generate;

import org.apache.commons.math.distribution.PoissonDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PoissonSamplerTest {
  @Test
  public void testBasics() {
    for (double alpha : new double[]{0.1, 1, 10, 100}) {
      checkDistribution(new PoissonSampler(alpha), alpha);
    }
  }

  private void checkDistribution(PoissonSampler pd, double alpha) {
    int[] count = new int[(int) Math.max(10, 5 * alpha)];
    for (int i = 0; i < 10000; i++) {
      count[pd.sample()]++;
    }

    PoissonDistribution ref = new PoissonDistributionImpl(alpha);
    for (int i = 0; i < count.length; i++) {
      assertEquals(ref.probability(i), count[i] / 10000.0, 2e-2);
    }
  }
}
