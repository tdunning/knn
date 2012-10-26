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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.search.Searcher;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.WeightedThing;

import java.util.*;

public class StreamingKMeans {
  // this parameter should be greater than 1, but not too much greater.
  // keeping BETA small makes the characteristic size grow more slowly
  // and small values of characteristic size seem to make the clustering
  // a bit better.  Too small a value of BETA, however, means that we
  // have to collapse the set of centroids too often.
  private static final double BETA = 1.3;

  private static final int TOP_MAX_SIZE = 100;

  // this is the current value of the characteristic size.  Points
  // which are much closer than this to a centroid will stick to it
  // almost certainly.  Points further than this to any centroid will
  // form a new cluster.
  private double distanceCutoff;

  private DistanceMeasure distanceMeasure;

  private UpdatableSearcher centroids;

  private int maxClusters;

  private List<WeightedVector> top;

  public StreamingKMeans(int dimension, final DistanceMeasure distance, int maxClusters) {
    this(distance, new ProjectionSearch(dimension, distance, 10, 20), maxClusters);
  }

  public StreamingKMeans(final DistanceMeasure distance, UpdatableSearcher searcher, int maxClusters) {
  }

  public Searcher cluster(WeightedVector datapoint) {
    List<WeightedVector> data = Lists.newArrayList();
    data.add(datapoint);
    return cluster(data);
  }

  public Searcher cluster(Iterable<WeightedVector> data) {
    return clusterInternal(data, 1);
  }

  private void estimateCutoff(Iterable<WeightedVector> data) {
    Iterable<WeightedVector> newTop = Iterables.concat(Iterables.limit(data, TOP_MAX_SIZE), top);
    for (WeightedVector datapoint : data)
      if (top.size() <= TOP_MAX_SIZE)
        top.add(datapoint);

    // first we need to have a reasonable value for what a "small" distance is
    // so we find the shortest distance between any of the first hundred data points
    distanceCutoff = Double.POSITIVE_INFINITY;
    for (List<WeightedThing<WeightedVector>> distances :
        new BruteSearch(newTop).search(newTop,  2)) {
      if (distances.size() > 1) {
        final double x = distances.get(1).getWeight();
        if (x != 0 && x < distanceCutoff) {
          distanceCutoff = x;
        }
      }
    }
  }

  private UpdatableSearcher clusterInternal(Iterable<WeightedVector> data, int depth) {

    // to cluster, we scan the data and either add each point to the nearest group or create a new group.
    // when we get too many groups, we need to increase the threshold and rescan our current groups
    Random rand = RandomUtils.getRandom();
    int n = 0;
    centroids.add(Centroid.create(0, Iterables.get(data, 0)));

    for (WeightedVector row : Iterables.skip(data, 1)) {
      // estimate distance d to closest centroid
      WeightedVector closest = centroids.search(row, 1).get(0).getValue();

      if (rand.nextDouble() < closest.getWeight() / distanceCutoff) {
        // add new centroid, note that the vector is copied because we may mutate it later
        centroids.add(Centroid.create(centroids.size(), row));
      } else {
        // merge against existing
        Centroid c = (Centroid) closest.getVector();
        centroids.remove(c, 1e-7);
        c.update(row);
        centroids.add(c);
      }

      if (depth < 2 && centroids.size() > maxClusters) {
        maxClusters = (int) Math.max(maxClusters, 10 * Math.log(n));
        // TODO does shuffling help?
        List<WeightedVector> shuffled = Lists.newArrayList(centroids);
        Collections.shuffle(shuffled);
        centroids = clusterInternal(shuffled, depth + 1);

        // in the original algorithm, with distributions with sharp scale effects, the
        // distanceCutoff can grow to excessive size leading sub-clustering to collapse
        // the centroids set too much. This test prevents increase in distanceCutoff
        // the current value is doing fine at collapsing the clusters.
        if (centroids.size() > 0.2 * maxClusters) {
          distanceCutoff *= BETA;
        }
      }
      n++;
    }
    return centroids;
  }
}

