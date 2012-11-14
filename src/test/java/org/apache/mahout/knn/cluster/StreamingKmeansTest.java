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
import com.sun.istack.internal.Nullable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.search.*;
// import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.List;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;
import static org.junit.runners.Parameterized.Parameters;


@RunWith(value = Parameterized.class)
public class StreamingKMeansTest {
  private static final int NUM_DATA_POINTS = 100000;
  private static final int NUM_DIMENSIONS = 3;
  private static final int NUM_PROJECTIONS = 4;
  private static final int SEARCH_SIZE = 10;

  private static Pair<List<Centroid>, List<Centroid>> syntheticData =
      DataUtils.sampleMultiNormalHypercube(NUM_DIMENSIONS, NUM_DATA_POINTS);

  private UpdatableSearcher searcher;

  public StreamingKMeansTest(UpdatableSearcher searcher) {
    this.searcher = searcher;
  }

  @Parameters
  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][] {
        {new ProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE)},
        {new FastProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE)},
        {new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), SEARCH_SIZE)}}
    );
  }

  @Test
  public void testClustering() {
    StreamingKMeans clusterer =
        new StreamingKMeans(searcher, 1 << NUM_DIMENSIONS,
            DataUtils.estimateDistanceCutoff(syntheticData.getFirst()));
    long startTime = System.currentTimeMillis();
    clusterer.cluster(syntheticData.getFirst());
    long endTime = System.currentTimeMillis();

    System.out.printf("Total number of clusters %d\n", clusterer.getCentroids().size());

    assertEquals("Total weight not preserved", totalWeight(clusterer.getCentroids()),
        totalWeight(syntheticData.getFirst()), 1e-9);

    // and verify that each corner of the cube has a centroid very nearby
    for (Vector mean : syntheticData.getSecond()) {
      WeightedThing<Vector> v = searcher.search(mean, 1).get(0);
      assertTrue(v.getWeight() < 0.05);
    }
    double clusterTime = (endTime - startTime) / 1000.0;
    System.out.printf("%s\n%.2f for clustering\n%.1f us per row\n\n",
        searcher.getClass().getName(), clusterTime,
        clusterTime / syntheticData.getFirst().size() * 1e6);

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
