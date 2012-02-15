package org.apache.mahout.knn.generate;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.NormalDistribution;
import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class NormalTest {
  @Test
  public void testSample() throws MathException {
    double[] data = new double[10001];
    Sampler<Double> sampler = new Normal();
    for (int i = 0; i < 10001; i++) {
      data[i] = sampler.sample();
    }
    Arrays.sort(data);

    NormalDistribution reference = new NormalDistributionImpl();

    Assert.assertEquals("Median", reference.inverseCumulativeProbability(0.5), data[5000], 0.02);
  }
}
