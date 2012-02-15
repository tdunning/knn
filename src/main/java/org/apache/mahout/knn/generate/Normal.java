package org.apache.mahout.knn.generate;

import java.util.Random;

public class Normal implements Sampler<Double> {
  private Random rand = new Random();

  public Double sample() {
    return rand.nextGaussian();
  }
}
