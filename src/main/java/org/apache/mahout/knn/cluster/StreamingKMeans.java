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
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Queues;
import com.sun.istack.internal.Nullable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.knn.search.*;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.WeightedThing;

import java.util.*;

public class StreamingKMeans {
  private double beta;

  private double clusterLogFactor;

  private double clusterOvershoot;

  private int estimatedNumClusters;

  private UpdatableSearcher centroids;

  // this is the current value of the characteristic size.  Points
  // which are much closer than this to a centroid will stick to it
  // almost certainly. Points further than this to any centroid will
  // form a new cluster.
  private double distanceCutoff = 10e-4;


  private int numProcessedDatapoints = 0;

  /**
   * Calls StreamingKMeans(searcher, estimatedNumClusters, initialDistanceCutoff, 1.3, 10, 0.2).
   * @see StreamingKMeans#StreamingKMeans(org.apache.mahout.knn.search.UpdatableSearcher, int, double, double, double, double)
   */
  public StreamingKMeans(UpdatableSearcher searcher, int estimatedNumClusters,
                         double initialDistanceCutoff) {
    this(searcher, estimatedNumClusters, initialDistanceCutoff, 1.3, 10, 0.2);
  }

  /**
   * Creates a new StreamingKMeans class given a searcher and the number of clusters to generate.
   *
   * @param searcher A Searcher that is used for performing nearest neighbor search. It MUST BE
   *                 EMPTY initially because it will be used to keep track of the cluster
   *                 centroids.
   * @param estimatedNumClusters An estimated number of clusters to generate for the data points.
   *                             This can adjusted, but the actual number will depend on the data.
   * @param initialDistanceCutoff The initial distance cutoff representing the value of the
   *                              distance between a point and its closest centroid after which
   *                              the new point will certainly be assigned to a new cluster.
   *                              @see DataUtils#estimateDistanceCutoff(Iterable,
   *                              org.apache.mahout.common.distance.DistanceMeasure,
   *                              int) for a rough estimate.
   * @param beta
   * @param clusterLogFactor
   * @param clusterOvershoot
   */
  public StreamingKMeans(UpdatableSearcher searcher, int estimatedNumClusters,
                         double initialDistanceCutoff, double beta, double clusterLogFactor,
                         double clusterOvershoot) {
    this.centroids = searcher;
    this.estimatedNumClusters = estimatedNumClusters;
    this.distanceCutoff = initialDistanceCutoff;
    this.beta = beta;
    this.clusterLogFactor = clusterLogFactor;
    this.clusterOvershoot = clusterOvershoot;
  }

  public UpdatableSearcher getCentroids() {
    return centroids;
  }

  public Iterable<Centroid> getCentroidsIterable() {
    return Iterables.transform(centroids, new Function<Vector, Centroid>() {
      @Override
      public Centroid apply(@Nullable Vector input) {
        return (Centroid)input;
      }
    });
  }

  // We can assume that for normal rows of a matrix, their weights are 1 because they represent
  // an individual vector.
  public UpdatableSearcher cluster(Matrix data) {
    return cluster(Iterables.transform(data, new Function<MatrixSlice, Centroid>() {
      @Override
      public Centroid apply(@Nullable MatrixSlice input) {
        // The key in a Centroid is actually the MatrixSlice's index.
        return Centroid.create(input.index(), input.vector());
      }
    }));
  }

  public UpdatableSearcher cluster(Iterable<Centroid> datapoints) {
    return clusterInternal(datapoints, false);
  }

  public UpdatableSearcher cluster(final Centroid v) {
    return cluster(new Iterable<Centroid>() {
      @Override
      public Iterator<Centroid> iterator() {
        return new Iterator<Centroid>() {
          private boolean accessed = false;

          @Override
          public boolean hasNext() {
            return !accessed;
          }

          @Override
          public Centroid next() {
            accessed = true;
            return v;
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
      }
    });
  }

  public int getEstimatedNumClusters() {
    return estimatedNumClusters;
  }

  public void setEstimatedNumClusters(int estimatedNumClusters) {
    this.estimatedNumClusters = estimatedNumClusters;
  }

  private UpdatableSearcher clusterInternal(Iterable<Centroid> datapoints,
                                            boolean collapseClusters) {
    // We clear the centroids we have in case of cluster collapse, the old clusters are the
    // datapoints but we need to re-cluster them.
    if (collapseClusters) {
      centroids.clear();
    }

    int numCentroidsToSkip = 0;
    if (centroids.size() == 0) {
      // Assign the first datapoint to the first cluster.
      // Adding a vector to a searcher would normally just reference the copy,
      // but we could potentially mutate it and so we need to make a clone.
      centroids.add(Iterables.get(datapoints, 0).clone());
      numCentroidsToSkip = 1;
      ++numProcessedDatapoints;
    }

    Random rand = RandomUtils.getRandom();
    // To cluster, we scan the data and either add each point to the nearest group or create a new group.
    // when we get too many groups, we need to increase the threshold and rescan our current groups
    for (WeightedVector row : Iterables.skip(datapoints, numCentroidsToSkip)) {
      // Get the closest vector and its weight as a WeightedThing<Vector>.
      // The weight of the WeightedThing is the distance to the query and the value is a
      // reference to one of the vectors we added to the searcher previously.
      WeightedThing<Vector> closestPair = centroids.search(row, 1).get(0);

      // We get a uniformly distributed random number between 0 and 1 and compare it with the
      // distance to the closest cluster divided by the distanceCutoff.
      // This is so that if the closest cluster is further than distanceCutoff,
      // closestPair.getWeight() / distanceCutoff > 1 which will trigger the creation of a new
      // cluster anyway.
      // However, if the ratio is less than 1, we want to create a new cluster with probability
      // proportional to the distance to the closest cluster.
      if (rand.nextDouble() < closestPair.getWeight() / distanceCutoff) {
        // Add new centroid, note that the vector is copied because we may mutate it later.
        centroids.add(row.clone());
      } else {
        // Merge the new point with the existing centroid. This will update the centroid's actual
        // position.
        // We know that all the points we inserted in the centroids searcher are (or extend)
        // WeightedVector, so the cast will always succeed.
        Centroid centroid = (Centroid)closestPair.getValue();
        // We will update the centroid by removing it from the searcher and reinserting it to
        // ensure consistency.
        if (!centroids.remove(centroid, 1e-7)) {
          throw new RuntimeException("Unable to remove centroid");
        }
        centroid.update(row);
        centroids.add(centroid);
      }

      if (!collapseClusters && centroids.size() > estimatedNumClusters) {
        estimatedNumClusters = (int) Math.max(estimatedNumClusters,
            clusterLogFactor * Math.log(numProcessedDatapoints));

        // TODO does shuffling help?
        List<Centroid> shuffled = Lists.newArrayList();
        for (Vector v : centroids) {
          shuffled.add((Centroid)v);
        }
        Collections.shuffle(shuffled);
        // Re-cluster using the shuffled centroids as data points. The centroids member variable
        // is modified directly.
        clusterInternal(shuffled, true);

        // In the original algorithm, with distributions with sharp scale effects, the
        // distanceCutoff can grow to excessive size leading sub-clustering to collapse
        // the centroids set too much. This test prevents increase in distanceCutoff if
        // the current value is doing well at collapsing the clusters.
        if (centroids.size() > clusterOvershoot * estimatedNumClusters) {
          distanceCutoff *= beta;
        }
      }
      if (!collapseClusters) {
        ++numProcessedDatapoints;
      }
    }

    // Normally, iterating through the searcher produces Vectors,
    // but since we always used Centroids, we adapt the return type.
    return centroids;
  }
}

