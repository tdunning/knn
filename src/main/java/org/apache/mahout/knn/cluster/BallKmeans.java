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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.sun.istack.internal.Nullable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
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
  // The searcher containing the centroids.
  private UpdatableSearcher centroids;

  // The number of clusters to cluster the data into.
  private int numClusters;

  // The maximum number of iterations of the algorithm to run waiting for the cluster assignments
  // to stabilize. If there are no changes in cluster assignment earlier, we can finish early.
  private int maxNumIterations;

  // When deciding which points to include in the new centroid calculation,
  // it's preferable to exclude outliers since it increases the rate of convergence.
  // So, we calculate the distance from each cluster to its closest neighboring cluster. When
  // evaluating the points assigned to a cluster, we compare the distance between the centroid to
  // the point with the distance between the centroid and its closest centroid neighbor
  // multiplied by this trimFraction. If the distance between the centroid and the point is
  // greater, we consider it an outlier and we don't use it.
  private double trimFraction;

  public BallKMeans(UpdatableSearcher searcher, int numClusters, int maxNumIterations) {
    this(searcher, numClusters, maxNumIterations, 0.9);
  }

  public BallKMeans(UpdatableSearcher searcher, int numClusters, int maxNumIterations,
                    double trimFraction) {
    Preconditions.checkArgument(searcher.size() == 0, "Searcher must be empty initially to " +
        "populate with centroids");
    Preconditions.checkArgument(numClusters > 0, "The requested number of clusters must be " +
        "positive");
    Preconditions.checkArgument(maxNumIterations > 0, "The maximum number of iterations must be " +
        "positive");
    this.centroids = searcher;
    this.numClusters = numClusters;
    this.maxNumIterations = maxNumIterations;
    this.trimFraction = trimFraction;
  }

  public UpdatableSearcher cluster(List<? extends WeightedVector> datapoints) {
    // use k-means++ to set initial centroids
    initializeSeeds(datapoints);
    // do k-means iterations with trimmed mean computation (aka ball k-means)
    iterativeAssignment(datapoints);
    return centroids;
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
   * @param datapoints The datapoints to select from.  These datapoints should be WeightedVectors of some kind.
   */
  private void initializeSeeds(List<? extends WeightedVector> datapoints) {
    Preconditions.checkArgument(datapoints.size() > 1, "Must have at least two datapoints points to cluster " +
        "sensibly");
    // Compute the centroid of all of the datapoints.  This is then used to compute the squared radius of the datapoints.
    Centroid center = new Centroid(datapoints.iterator().next());
    for (WeightedVector row : Iterables.skip(datapoints, 1)) {
      center.update(row);
    }
    // Given the centroid, we can compute \Delta_1^2(X), the total squared distance for the datapoints
    // this accelerates seed selection.
    double radius = 0;
    DistanceMeasure l2 = new SquaredEuclideanDistanceMeasure();
    for (WeightedVector row : datapoints) {
      radius += l2.distance(row, center);
    }

    // Find the first seed c_1 (and conceptually the second, c_2) as might be done in the 2-means clustering so that
    // the probability of selecting c_1 and c_2 is proportional to || c_1 - c_2 ||^2.  This is done
    // by first selecting c_1 with probability:
    //
    // p(c_1) = sum_{c_1} || c_1 - c_2 ||^2 \over sum_{c_1, c_2} || c_1 - c_2 ||^2
    //
    // This can be simplified to:
    //
    // p(c_1) = \Delta_1^2(X) + n || c_1 - c ||^2 / (2 n \Delta_1^2(X))
    //
    // where c = \sum x / n and \Delta_1^2(X) = sum || x - c ||^2
    //
    // All subsequent seeds c_i (including c_2) can then be selected from the remaining points with probability
    // proportional to Pr(c_i == x_j) = min_{m < i} || c_m - x_j ||^2.

    // Multinomial distribution of vector indices for the selection seeds. These correspond to
    // the indices of the vectors in the original datapoints list.
    Multinomial<Integer> seedSelector = new Multinomial<Integer>();
    for (int i = 0; i < datapoints.size(); ++i) {
      double selectionProbability =
          radius + datapoints.size() * l2.distance(datapoints.get(i), center);
      seedSelector.add(i, selectionProbability);
    }

    Centroid c_1 = new Centroid((WeightedVector)datapoints.get(seedSelector.sample()).clone());
    c_1.setIndex(0);
    // Construct a set of weighted things which can be used for random selection.  Initial weights are
    // set to the squared distance from c_1
    for (int i = 0; i < datapoints.size(); ++i) {
      WeightedVector row = datapoints.get(i);
      final double w = l2.distance(c_1, row) * row.getWeight();
      seedSelector.set(i, w);
    }

    // From here, seeds are selected with probablity proportional to:
    //
    // r_i = min_{c_j} || x_i - c_j ||^2
    //
    // when we only have c_1, we have already set these distances and as we select each new
    // seed, we update the minimum distances.
    centroids.add(c_1);
    int clusterIndex = 1;
    while (centroids.size() < numClusters) {
      // Select according to weights.
      int seedIndex = seedSelector.sample();
      Centroid nextSeed = new Centroid((WeightedVector)datapoints.get(seedIndex).clone());
      nextSeed.setIndex(clusterIndex++);
      centroids.add(nextSeed);
      // Don't select this one again.
      seedSelector.set(seedIndex, 0);
      // Re-weight everything according to the minimum distance to a seed.
      for (int currSeedIndex : seedSelector) {
        WeightedVector curr = datapoints.get(currSeedIndex);
        double newWeight = nextSeed.getWeight() * l2.distance(nextSeed, curr);
        if (newWeight < seedSelector.getWeight(currSeedIndex)) {
          seedSelector.set(currSeedIndex, newWeight);
        }
      }
    }
  }

  /**
   * Examines the datapoints and updates cluster centers to be the centroid of the nearest datapoints points.  To
   * compute a new center for cluster c_i, we average all points that are closer than d_i * trimFraction
   * where d_i is
   * <p/>
   * d_i = min_j \sqrt ||c_j - c_i||^2
   * <p/>
   * By ignoring distant points, the centroids converge more quickly to a good approximation of the
   * optimal k-means solution (given good starting points).
   *
   * @param datapoints          Rows containing WeightedVectors
   */
  private void iterativeAssignment(List<? extends WeightedVector> datapoints) {
    DistanceMeasure l2 = new EuclideanDistanceMeasure();
    // closestClusterDistances.get(i) is the distance from the i'th cluster to its closest
    // neighboring cluster.
    List<Double> closestClusterDistances = Lists.newArrayListWithExpectedSize(numClusters);
    // clusterAssignments[i] == j means that the i'th point is assigned to the j'th cluster. When
    // these don't change, we are done.
    List<Integer> clusterAssignments = Lists.newArrayListWithExpectedSize(datapoints.size());
    // Each point is assigned to the invalid "-1" cluster initially.
    for (int i = 0; i < datapoints.size(); ++i) {
      clusterAssignments.add(-1);
    }

    boolean changed = true;
    for (int i = 0; changed && i < maxNumIterations; i++) {
      // We compute what the distance between each cluster and its closest neighbor is to set a
      // proportional distance threshold for points that should be involved in calculating the
      // centroid.
      closestClusterDistances.clear();
      for (Vector center : centroids) {
        Vector closestOtherCluster = centroids.search(center, 2).get(1).getValue();
        closestClusterDistances.add(l2.distance(center, closestOtherCluster));
      }

      // Copies the current cluster centroids to newClusters and sets their weights to 0. This is
      // so we calculate the new centroids as we go through the datapoints.
      List<Centroid> newCentroids = Lists.newArrayList();
      for (Vector centroid : centroids) {
        // need a deep copy because we will mutate these values
        Centroid newCentroid = (Centroid)centroid.clone();
        newCentroid.setWeight(0);
        newCentroids.add(newCentroid);
      }

      // Pass over the datapoints computing new centroids.
      for (int j = 0; j < datapoints.size(); ++j) {
        WeightedVector datapoint = datapoints.get(j);
        // Get the closest cluster this point belongs to.
        WeightedThing<Vector> closestPair = centroids.search(datapoint, 1).get(0);
        int closestIndex = ((WeightedVector)closestPair.getValue()).getIndex();
        double closestDistance = closestPair.getWeight();
        // Update its cluster assignment if necessary.
        if (closestIndex != clusterAssignments.get(j)) {
          changed = true;
          clusterAssignments.set(j, closestIndex);
        }
        // Only update if the datapoints point is near enough. What this means is that the weight
        // of outliers is NOT taken into account and the final weights of the centroids will
        // reflect this (it will be less or equal to the initial sum of the weights).
        if (closestDistance < closestClusterDistances.get(closestIndex) *  trimFraction) {
          newCentroids.get(closestIndex).update(datapoint);
        }
      }
      // Add new centers back into searcher.
      centroids.clear();
      centroids.addAll(newCentroids);
    }
  }

  @Override
  public Iterator<Centroid> iterator() {
    return Iterators.transform(centroids.iterator(), new Function<Vector, Centroid>() {
      @Override
      public Centroid apply(@Nullable Vector input) {
        Preconditions.checkArgument(input instanceof Centroid, "Non-centroid in centroids " +
            "searcher");
        return (Centroid)input;
      }
    });
  }
}
