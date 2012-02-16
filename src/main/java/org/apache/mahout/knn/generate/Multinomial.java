package org.apache.mahout.knn.generate;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Samples from a multinomial distribution using a fast tree algorithm.
 */
public class Multinomial<T> implements Sampler<T> {
  private SearchTree<T> tree;
  private Random rand = new Random();
  private static final double EPSILON = 1e-10;

  public Multinomial(Multiset<T> counts, int width) {
    Preconditions.checkArgument(counts.size() > 0, "Need some data to build sampler");
    List<WeightedThing<T>> things = Lists.newArrayList();
    double n = counts.size();
    for (T t : counts.elementSet()) {
      things.add(new WeightedThing<T>(t, counts.count(t) / n));
    }
    Collections.sort(things);

    // now convert to cumulative weights to help with encoding as a tree
    double sum = 0;
    for (WeightedThing<T> thing : things) {
      final double w = thing.weight;
      sum += w;
      if (sum > 1) {
        // only can happen with round-off errors.  Since we add numbers up smallest
        // first, this should be a very minor probability.
        sum = 1;
      }
      thing.weight = sum;
    }
    // avoid round-off errors
    things.get(things.size() - 1).weight = 1;

    // this allows us to build a tree that will help us sample fast
    tree = buildTree(0, 1, things, width);
  }

  /**
   * Recursively builds a search tree.
   *
   * @param low    The low bound for the search for this tree
   * @param high   The high bound for the search of this tree
   * @param things A list of things to be included in this branch of the tree
   * @param width  Branching factor for the tree
   * @return A search tree which may be an interior node or a sub-tree.
   */
  private SearchTree<T> buildTree(double low, double high, List<WeightedThing<T>> things, int width) {
    Preconditions.checkArgument(things.size() > 0, "Can't construct a tree with nothing");
    Preconditions.checkArgument(low <= things.get(0).weight, "First element is outside outside of correct range");
    Preconditions.checkArgument(high <= things.get(things.size() - 1).weight, "Last element is outside of correct range");

    if (things.size() == 1) {
      return new Leaf<T>(things.get(0).value);
    } else if (things.size() == 2) {
      final WeightedThing<T> t0 = things.get(0);
      final WeightedThing<T> t1 = things.get(1);
      return new Triplet<T>(ImmutableList.of(t0.value, t1.value), t0.weight, high + 1);
    } else if (things.size()==3){
      final WeightedThing<T> t0 = things.get(0);
      final WeightedThing<T> t1 = things.get(1);
      final WeightedThing<T> t2 = things.get(2);
      return new Triplet<T>(ImmutableList.of(t0.value, t1.value, t2.value), t0.weight, t1.weight);
    } else if (things.size() <= width && high - low < EPSILON) {
      // these items are squeezed into such a small space that we really don't have to
      // worry about the details.  Thus we just give them all equal (and very small)
      // probabilities.
      Node<T> r = new Node<T>();
      for (WeightedThing<T> thing : things) {
        r.add(new Leaf<T>(thing.value));
      }
      return r;
    } else {
      // each sub-tree here will take a uniform chunk of probability space.
      // if that chunk has only one element in it, that element will be a leaf
      int base = 0;
      Node<T> r = new Node<T>();
      final double step = (high - low) / width;
      for (int i = 0; i < width; i++) {
        double cutoff = Math.min(1, low + step);
        int top = base;
        while (top < things.size() && things.get(top).weight < cutoff) {
          top++;
        }
        r.add(buildTree(low, cutoff, things.subList(base, top + 1), width));
        low = cutoff;
        base = top;
      }
      return r;
    }
  }

  public T sample() {
    final double p = rand.nextDouble();
    return sample(p);
  }

  public T sample(double p) {
    return tree.find(p);
  }

  private static interface SearchTree<T> {
    T find(double p);
  }

  private static class Node<T> implements SearchTree<T> {
    List<SearchTree<T>> children;

    public Node() {
      children = Lists.newArrayList();
    }

    public void add(SearchTree<T> node) {
      children.add(node);
    }

    public T find(double p) {
      if (p < 0) {
        p = 0;
      }
      if (p > 1) {
        p = 1;
      }
      int slot = (int) (p * children.size());
      if (slot == children.size()) {
        slot = slot - 1;
      }
      return children.get(slot).find(p);
    }
  }

  private static class Triplet<T> implements SearchTree<T> {
    double p1, p2;
    List<T> values;

    private Triplet(List<T> values, double p1, double p2) {
      this.values = values;
      this.p2 = p2;
      this.p1 = p1;
    }

    public T find(double p) {
      if (p < p1) {
        return values.get(0);
      } else if (p >= p2) {
        return values.get(2);
      } else {
        return values.get(1);
      }
    }
  }

  private static class Leaf<T> implements SearchTree<T> {
    T value;

    public Leaf(T value) {
      this.value = value;
    }

    public T find(double p) {
      return value;
    }
  }

  private static class WeightedThing<T> implements Comparable<WeightedThing<T>> {
    double weight;
    T value;

    public WeightedThing(T thing, double weight) {
      this.value = thing;
      this.weight = weight;
    }

    public int compareTo(WeightedThing<T> other) {
      return Double.compare(this.weight, other.weight);
    }
  }
}
