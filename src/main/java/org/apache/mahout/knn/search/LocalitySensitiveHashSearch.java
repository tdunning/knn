package org.apache.mahout.knn.search;

import com.google.common.base.Function;
import com.google.common.collect.*;
import com.sun.istack.internal.Nullable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.*;
import org.apache.mahout.math.jet.random.Normal;
import org.apache.mahout.math.random.WeightedThing;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Implements a Searcher that uses locality sensitivity hash as a first pass approximation
 * to estimate distance without floating point math.  The clever bit about this implementation
 * is that it does an adaptive cutoff for the cutoff on the bitwise distance.  Making this
 * cutoff adaptive means that we only needs to make a single pass through the data.
 */
public class LocalitySensitiveHashSearch extends UpdatableSearcher implements Iterable<Vector> {
  private static final int BITS = 64;
  @SuppressWarnings("PointlessBitwiseExpression")
  private static final long BITMASK = -1L >>> 64 - BITS;

  private Set<HashedVector> trainingVectors = Sets.newHashSet();


  // this matrix of 32 random vectors is used to compute the Locality Sensitive Hash
  // we compute the dot product with these vectors using a matrix multiplication and then use just
  // sign of each result as one bit in the hash
  private Matrix projection;

  // the search size determines how many top results we retain.  We do this because the hash distance
  // isn't guaranteed to be entirely monotonic with respect to the real distance.  To the extent that
  // actual distance is well approximated by hash distance, then the searchSize can be decreased to
  // roughly the number of results that you want.
  private int searchSize;

  // controls how the hash limit is raised.  0 means use minimum of distribution, 1 means use first quartile.
  // intermediate values indicate an interpolation should be used.  Negative values mean to never increase.
  private double hashLimitStrategy = 0.9;

  private int distanceEvaluations = 0;

  public LocalitySensitiveHashSearch(DistanceMeasure distanceMeasure, int numDimensions,
                                     int searchSize) {
    super(distanceMeasure);
    this.searchSize = searchSize;

    projection = new DenseMatrix(BITS, numDimensions);
    projection.assign(new Normal(0, 1, RandomUtils.getRandom()));
  }

  @Override
  public List<WeightedThing<Vector>> search(Vector q, int numberOfNeighbors) {
    long queryHash = HashedVector.computeHash64(q, projection);

    // we keep an approximation of the closest vectors here
    PriorityQueue<WeightedThing<Vector>> top = new
        PriorityQueue<WeightedThing<Vector>>(getSearchSize(),
        Ordering.natural().reverse());

    // we keep the counts of the hash distances here.  This lets us accurately
    // judge what hash distance cutoff we should use.
    int[] hashCounts = new int[BITS + 1];

    // we scan the vectors using bit counts as an approximation of the dot product so we can do as few
    // full distance computations as possible.  Our goal is to only do full distance computations for
    // vectors with hash distance at most as large as the searchSize biggest hash distance seen so far.

    // in this loop, we have the invariants that
    //
    // limitCount = sum_{i<hashLimit} hashCount[i]
    // and
    // limitCount >= searchSize && limitCount - hashCount[hashLimit-1] < searchSize

    OnlineSummarizer[] distribution = new OnlineSummarizer[BITS + 1];
    for (int i = 0; i < BITS + 1; i++) {
      distribution[i] = new OnlineSummarizer();
    }

    int hashLimit = BITS;
    int limitCount = 0;
    double distanceLimit = Double.POSITIVE_INFINITY;
    for (HashedVector v : trainingVectors) {
      int bitDot = Long.bitCount(v.getHash() ^ queryHash);
      if (bitDot <= hashLimit) {
        distanceEvaluations++;
        double d = distanceMeasure.distance(q, v);
        distribution[bitDot].add(d);
        if (d < distanceLimit) {
          top.add(new WeightedThing<Vector>(v, d));
          while (top.size() > searchSize) {
            top.poll();
          }

          if (top.size() == searchSize) {
            distanceLimit = top.peek().getWeight();
          }

          hashCounts[bitDot]++;
          limitCount++;
          while (hashLimit > 0 && limitCount - hashCounts[hashLimit - 1] > searchSize) {
            hashLimit--;
            limitCount -= hashCounts[hashLimit];
          }

          if (hashLimitStrategy >= 0) {
            while (hashLimit < 32 && distribution[hashLimit].getCount() > 10 &&
                (hashLimitStrategy * distribution[hashLimit].getQuartile(1)) + ((1 - hashLimitStrategy) * distribution[hashLimit].getQuartile(0)) < distanceLimit) {
              limitCount += hashCounts[hashLimit];
              hashLimit++;
            }
          }
        }
      }
    }

    List<WeightedThing<Vector>> r = Lists.newArrayList(Iterables.transform(top, new Function<WeightedThing<Vector>, WeightedThing<Vector>>() {
      @Override
      public WeightedThing<Vector> apply(@Nullable WeightedThing<Vector> input) {
        return new WeightedThing<Vector>(((HashedVector)(input.getValue())).getVector(),
            input.getWeight());
      }
    }));
    Collections.sort(r);
    return r.subList(0, numberOfNeighbors);
  }


  @Override
  public void add(Vector v) {
    trainingVectors.add(new HashedVector(v, projection, HashedVector.INVALID_INDEX, BITMASK));
  }


  public int size() {
    return trainingVectors.size();
  }

  public int getSearchSize() {
    return searchSize;
  }

  public void setSearchSize(int size) {
    searchSize = size;
  }

  public void setRaiseHashLimitStrategy(double strategy) {
    hashLimitStrategy = strategy;
  }

  public int resetEvaluationCount() {
    int r = distanceEvaluations;
    distanceEvaluations = 0;
    return r;
  }

  @Override
  public Iterator<Vector> iterator() {
    return new AbstractIterator<Vector>() {
      int index = 0;
      Iterator<HashedVector> data = trainingVectors.iterator();

      @Override
      protected Vector computeNext() {
        if (!data.hasNext()) {
          return endOfData();
        } else {
          return data.next().getVector();
        }
      }
    };
  }

  @Override
  public boolean remove(Vector v, double epsilon) {
    return trainingVectors.remove(
        new HashedVector(v, projection, HashedVector.INVALID_INDEX, BITMASK));
  }

  @Override
  public void clear() {
    trainingVectors.clear();
  }
}
