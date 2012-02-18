package org.apache.mahout.knn.generate;

/**
* Handy for creating multinomial distributions of things.
*/
final class WeightedThing<T> implements Comparable<WeightedThing<T>> {
  public double weight;
  public final T value;

  public WeightedThing(T thing, double weight) {
    this.value = thing;
    this.weight = weight;
  }

  public int compareTo(WeightedThing<T> other) {
    return Double.compare(this.weight, other.weight);
  }
}
