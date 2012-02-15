package org.apache.mahout.knn.generate;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.NormalDistribution;
import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class FooTest {
  @Test
  public void testSample() throws MathException {
    NormalDistribution dist = new NormalDistributionImpl();

    double[] data = new double[10001];
    Foo sampler = new Foo();

    for (int i = 0; i <= 10000; i++) {
      data[i] = sampler.sample();
    }
    Arrays.sort(data);

    Assert.assertEquals("Median", dist.inverseCumulativeProbability(0.5), data[5000], 0.01);
  }
}
