package org.apache.mahout.knn.generate;

/**
 * Produces samples from some specified type.
 */
public interface Sampler<T> {
  T sample();
}
