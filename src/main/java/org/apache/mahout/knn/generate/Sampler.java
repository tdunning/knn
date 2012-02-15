package org.apache.mahout.knn.generate;

/**
 * Samples from a generic type.
 */
public interface Sampler<T> {
  T sample();
}
