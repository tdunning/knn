package org.apache.mahout.knn.generate;

import java.util.Random;

/**
 * Generate some random numbers
 */
public class Foo implements Sampler<Double> {
  private Random rand = new Random();

  public Double sample() {
    return rand.nextGaussian();
  }
}
