/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.knn.cluster;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.Searcher;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.random.Multinomial;
import org.apache.mahout.math.random.WeightedThing;

import java.util.Iterator;
import java.util.List;

/**
 * Implements a ball k-means algorithm for weighted vectors with probabilistic seeding similar to k-means++.
 * The idea is that k-means++ gives good starting clusters and ball k-means can tune up the final result very nicely
 * in only a few passes (or even in a single iteration for well-clusterable data).
 * <p/>
 * A good reference for this class of algorithms is "The Effectiveness of Lloyd-Type Methods for the k-Means Problem"
 * by Rafail Ostrovsky, Yuval Rabani, Leonard J. Schulman and Chaitanya Swamy.  The code here uses the seeding strategy
 * as described in section 4.1.1 of that paper and the ball k-means step as described in section 4.2.  We support
 * multiple iterations in contrast to the algorithm described in the paper.
 */
public class BallKMeans implements Iterable<Centroid> {
  private Searcher centroidFinder;
  private List<Centroid> clusters = Lists.newArrayList();

  public BallKMeans(int k, Iterable<? extends WeightedVector> data, int maxIterations) {
    DistanceMeasure metric = new EuclideanDistanceMeasure();

    centroidFinder = new BruteSearch(metric);

    // use k-means++ to set initial centroids
    initializeSeeds(k, data);

    // do k-means iterations with trimmed mean computation (aka ball k-means)
    iterativeAssignment(data, maxIterations, 0.9);
  }

  /**
   * Selects some of the original points according to the k-means++ algorithm.  The basic idea is that
   * points are selected with probability proportional to their distance from any selected point.  In
   * this version, points have weights which multiply their likelihood of being selected.  This is the
   * same as if there were as many copies of the same point as indicated by the weight.
   * <p/>
   * This is pretty expensive, but it vastly improves the quality and convergences of the k-means algorithm.
   * The basic idea can be made much faster by only processing a random subset of the original points.
   * In the context of streaming k-means, the total number of possible seeds will be about k log n so this
   * selection will cost O(k^2 (log n)^2) which isn't much worse than the random sampling idea.  At
   * n = 10^9, the cost of this initialization will be about 10x worse than a reasonable random sampling
   * implementation.
   * <p/>
   * The side effect of this method is to fill the centroids structure.
   * itself.
   *
   * @param k    The number of points to select.
   * @param data The data to select from.  These data should be WeightedVectors of some kind.
   */
  private void initializeSeeds(int k, Iterable<? extends WeightedVector> data) {
    // ensure that we have at least a few rows.  The real value of n will be computed shortly.
    int n = 0;
    for (WeightedVector row : data) {
      if (n > 3) break;
      n++;
    }
    Preconditions.checkArgument(n >= 2, "Must have at least two data points to cluster sensibly");

    // Compute the centroid of all of the data.  This is then used to compute the squared radius of the data.
    Centroid center = new Centroid(data.iterator().next());
    for (WeightedVector row : Iterables.skip(data, 1)) {
      center.update(row);
    }
    /*
    Vector center = data.iterator().next().vector().like();
    n = 0;
    for (WeightedVector row : data) {
      final double w = row.getWeight();
      n += w;
      center.assign(row, Functions.plusMult(w));
    }
    center.assign(Functions.div(n));
    */

    // given the centroid, we can compute \Delta_1^2(X), the total squared distance for the data
    // this accelerates seed selection
    double radius = 0;
    DistanceMeasure l2 = new SquaredEuclideanDistanceMeasure();

    for (WeightedVector row : data) {
      radius += l2.distance(row, center);
    }

    // Find the first seed c_1 (and conceptually the second, c_2) as might be done in the 2-means clustering so that
    // the probability of selecting c_1 and c_2 is proportional to || c_1 - c_2 ||^2.  This is done
    // by first selecting c_1 with probability
    //
    // p(c_1) = sum_{c_1} || c_1 - c_2 ||^2 \over sum_{c_1, c_2} || c_1 - c_2 ||^2
    //
    // This can be simplified to
    //
    // p(c_1) = \Delta_1^2(X) + n || c_1 - c ||^2 / (2 n \Delta_1^2(X))
    //
    // where c = \sum x / n and \Delta_1^2(X) = sum || x - c ||^2
    //
    // All subsequent seeds c_i (including c_2) can then be selected from the remaining points with probability
    // proportional to Pr(c_i == x_j) = min_{m < i} || c_m - x_j ||^2

    Multinomial<WeightedVector> seedSelector = new Multinomial<WeightedVector>();
    for (WeightedVector row : data) {
      double p = (radius + n * l2.distance(row, center));
      seedSelector.add(row, p);
    }
    WeightedVector c_1 = seedSelector.sample();

    // Construct a set of weighted things which can be used for random selection.  Initial weights are
    // set to the squared distance from c_1
    for (WeightedVector row : data) {
      final double w = l2.distance(c_1, row) * row.getWeight();
      seedSelector.set(row, w);
    }

    // From here, seeds are selected with probablity proportional to
    //
    // r_i = min_{c_j} || x_i - c_j ||^2
    //
    // when we only have c_1, we have already set these distances and as we select each new
    // seed, we update the minimum distances.

    List<WeightedVector> seeds = Lists.newArrayList();
    seeds.add(c_1);
    while (seeds.size() < k) {
      // select according to weights

      WeightedVector nextSeed = seedSelector.sample();

      seeds.add(nextSeed);

      // don't select this one again
      seedSelector.set(nextSeed, 0);

      // now re-weight everything according to the minimum distance to a seed
      for (WeightedVector s : seedSelector) {
        double newWeight = nextSeed.getWeight() * l2.distance(nextSeed, s);
        if (newWeight < seedSelector.getWeight(s)) {
          seedSelector.set(s, newWeight);
        }
      }
    }

    int i = 0;
    for (WeightedVector seed : seeds) {
      centroidFinder.add(new Centroid(i, seed.getVector(), 1));
      clusters.add(new Centroid(i, seed.getVector(), seed.getWeight()));
      i++;
    }
  }

  private WeightedVector asWeightedVector(MatrixSlice row) {
    WeightedVector v;
    if (row.vector() instanceof WeightedVector) {
      v = (WeightedVector) row.vector();
    }   else {
      v = new WeightedVector(row.vector(), 1, row.index());
    }
    return v;
  }

  /**
   * Examines the data and updates cluster centers to be the centroid of the nearest data points.  To
   * compute a new center for cluster c_i, we average all points that are closer than d_i * trimFraction
   * where d_i is
   * <p/>
   * d_i = min_j \sqrt ||c_j - c_i||^2
   * <p/>
   * By ignoring distant points, the centroids converge more quickly to a good approximation of the
   * optimal k-means solution (given good starting points).
   *
   * @param data          Rows containing WeightedVectors
   * @param maxIterations The maximum number of iterations to be performed.  This can be quite low in
   *                      contrast to most k-means implementations.
   * @param trimFaction   Controls which data points to be included in the center computation.
   *                      Typically trimFaction = 1.0/3 is used.  Trim fraction is applied to
   *                      the distance, not the number of data points.
   */
  private void iterativeAssignment(Iterable<? extends WeightedVector> data, int maxIterations,
                                   double trimFaction) {
    List<Double> d = Lists.newArrayList();
    DistanceMeasure l2 = new EuclideanDistanceMeasure();

    // holds previous cluster assignments ... when these don't change, we are done
    List<Integer> assignments = Lists.newArrayList();

    for (int i = 0; i < maxIterations; i++) {
      // need to know how the closest other cluster for each cluster
      d.clear();
      for (Vector center : centroidFinder) {
        Vector closestOtherCluster = centroidFinder.search(center, 2).get(1).getValue();
        d.add(l2.distance(center, closestOtherCluster));
      }

      boolean changed = false;

      // holds new cluster centers
      List<Centroid> newClusters = Lists.newArrayList();
      for (Centroid cluster : clusters) {
        // need a deep copy because we will mutate these values
        final Centroid c = new Centroid(cluster);
        c.setWeight(0);
        newClusters.add(c);
      }

      // pass over the data computing new centroids
      for (WeightedVector row : data) {
        while (assignments.size() <= row.getIndex()) {
          assignments.add(-1);
        }

        WeightedVector closest =
            (WeightedVector)centroidFinder.search(row, 1).get(0).getValue();
        if (closest.getIndex() != assignments.get(row.getIndex())) {
          changed = true;
        }
        assignments.set(row.getIndex(), closest.getIndex());

        // only update if the data point is near enough
        if (true || closest.getWeight() < d.get(closest.getIndex()) * trimFaction) {
          newClusters.get(closest.getIndex()).update(row);
        }
      }

      if (!changed) {
        break;
      }

      // add new centers back into searcher
      centroidFinder.clear();
      for (Centroid cluster : newClusters) {
        centroidFinder.add(cluster);
      }
    }
  }

  @Override
  public Iterator<Centroid> iterator() {
    return clusters.iterator();
  }
}
