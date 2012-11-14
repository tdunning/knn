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
import com.sun.istack.internal.Nullable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.SyntheticDataUtils;
import org.apache.mahout.knn.search.*;
// import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;


public class StreamingKMeansTest {
  private static int NUM_DATA_POINTS = 100000;
  private static int NUM_DIMENSIONS = 3;
  private static int NUM_PROJECTIONS = 4;
  private static int SEARCH_SIZE = 10;

  private Iterable<Centroid> getMatrixAsCentroids(Matrix m) {
    return Iterables.transform(m, new Function<MatrixSlice, Centroid>() {
      @Override
      public Centroid apply(@Nullable MatrixSlice input) {
        // The key in a Centroid is actually the MatrixSlice's index.
        return Centroid.create(input.index(), input.vector());
      }
    });
  }
  @Test
  public void testEstimateBeta() {
    Matrix m = new DenseMatrix(8, 3);
    for (int i = 0; i < 8; i++) {
      m.viewRow(i).assign(new double[]{0.125 * (i & 4), i & 2, i & 1});
    }
    StreamingKMeans clusterer = new StreamingKMeans(new BruteSearch(new EuclideanDistanceMeasure()), 100);
    Assert.assertEquals(0.5, clusterer.estimateDistanceCutoff(getMatrixAsCentroids(m)), 1e-9);
  }

  @Test
  public void testClustering() {
    Pair<List<Centroid>, List<Centroid>> syntheticClusters =
        SyntheticDataUtils.sampleMultiNormalHypercube(NUM_DIMENSIONS, NUM_DATA_POINTS);

    final EuclideanDistanceMeasure distanceMeasure = new EuclideanDistanceMeasure();
    clusterCheck(syntheticClusters, "projection", new ProjectionSearch(distanceMeasure,
        NUM_PROJECTIONS, SEARCH_SIZE));
    clusterCheck(syntheticClusters, "lsh", new LocalitySensitiveHashSearch(distanceMeasure,
        SEARCH_SIZE));
  }

  private void clusterCheck(Pair<List<Centroid>, List<Centroid>> syntheticData,
                            String title, UpdatableSearcher updatableSearcher) {
    long startTime = System.currentTimeMillis();
    StreamingKMeans clusterer = new StreamingKMeans(updatableSearcher, 1 << NUM_DIMENSIONS);
    clusterer.cluster(syntheticData.getFirst());
    long endTime = System.currentTimeMillis();

    System.out.printf("Total number of clusters %d\n", clusterer.getCentroids().size());

    assertEquals("Total weight not preserved", totalWeight(clusterer.getCentroids()),
        totalWeight(syntheticData.getFirst()), 1e-9);

    // and verify that each corner of the cube has a centroid very nearby
    for (Vector mean : syntheticData.getSecond()) {
      WeightedThing<Vector> v = updatableSearcher.search(mean, 1).get(0);
      assertTrue(v.getWeight() < 0.05);
    }
    double clusterTime = (endTime - startTime) / 1000.0;
    System.out.printf("%s\n%.2f for clustering\n%.1f us per row\n\n",
        title, clusterTime, clusterTime / syntheticData.getFirst().size() * 1e6);

    // verify that the total weight of the centroids near each corner is correct
    double[] cornerWeights = new double[1 << NUM_DIMENSIONS];
    Searcher trueFinder = new BruteSearch(new EuclideanDistanceMeasure());
    for (Vector trueCluster : syntheticData.getSecond()) {
      trueFinder.add(trueCluster);
    }
    for (Centroid centroid : clusterer.getCentroidsIterable()) {
      WeightedThing<Vector> closest = trueFinder.search(centroid, 1).get(0);
      cornerWeights[((Centroid)closest.getValue()).getIndex()] += centroid.getWeight();
    }
    int expectedNumPoints = NUM_DATA_POINTS / (1 << NUM_DIMENSIONS);
    for (double v : cornerWeights) {
      assertEquals(expectedNumPoints, v, 0);
    }
  }

  private double totalWeight(Iterable<? extends Vector> data) {
    double sum = 0;
    for (Vector row : data) {
      if (row instanceof WeightedVector) {
        sum += ((WeightedVector)row).getWeight();
      } else {
        sum++;
      }
    }
    return sum;
  }
}
