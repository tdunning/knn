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
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.search.*;
// import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Assert;
import org.junit.Test;

import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


public class StreamingKMeansTest {
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
    // construct data samplers centered on the corners of a unit cube
    Matrix mean = new DenseMatrix(8, 3);
    List<MultiNormal> rowSamplers = Lists.newArrayList();
    for (int i = 0; i < 8; i++) {
      mean.viewRow(i).assign(new double[]{0.25 * (i & 4), 0.5 * (i & 2), i & 1});
      rowSamplers.add(new MultiNormal(0.01, mean.viewRow(i)));
    }

    // sample a bunch of data points
    Matrix data = new DenseMatrix(100000, 3);
    for (MatrixSlice row : data) {
      row.vector().assign(rowSamplers.get(row.index() % 8).sample());
    }

    // cluster the data
    final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();
    clusterCheck(mean, "projection", data, new ProjectionSearch(distance, 3, 4, 10));
    clusterCheck(mean, "lsh", data, new LocalitySensitiveHashSearch(distance, 3, 10));
  }

  private void clusterCheck(Matrix mean, String title, Matrix data, UpdatableSearcher searcher) {
    long t0 = System.currentTimeMillis();
    StreamingKMeans clusterer = new StreamingKMeans(searcher, 1000);
    clusterer.cluster(getMatrixAsCentroids(data));
    long t1 = System.currentTimeMillis();

    System.out.printf("Number of clusters %d\n", clusterer.getCentroids().size());

    assertEquals("Total weight not preserved", totalWeightFromMatrixSlices(data),
        totalWeight(searcher), 1e-9);

    // and verify that each corner of the cube has a centroid very nearby
    for (MatrixSlice row : mean) {
      WeightedThing<Vector> v = searcher.search(row.vector(), 1).get(0);
      assertTrue(v.getWeight() < 0.05);
    }
    System.out.printf("%s\n%.2f for clustering\n%.1f us per row\n\n",
        title, (t1 - t0) / 1000.0, (t1 - t0) / 1000.0 / data.rowSize() * 1e6);

    // verify that the total weight of the centroids near each corner is correct
    double[] w = new double[8];
    Searcher trueFinder = new BruteSearch(new EuclideanDistanceMeasure());
    for (MatrixSlice trueCluster : mean) {
      trueFinder.add(
          new WeightedVector(trueCluster.vector(), 1, trueCluster.index()));
    }
    for (Centroid centroid : clusterer.getCentroidsIterable()) {
      WeightedThing<Vector> z = trueFinder.search(centroid, 1).get(0);
      w[((WeightedVector)(z.getValue())).getIndex()] += centroid.getWeight();
          //((WeightedVector)(z.getValue())).getWeight();
    }
    for (double v : w) {
      assertEquals(12500, v, 0);
    }
  }

  private double totalWeightFromMatrixSlices(Iterable<MatrixSlice> data) {

    double sum = 0;
    for (MatrixSlice row : data) {
      if (row.vector() instanceof WeightedVector) {
        sum += ((WeightedVector) row.vector()).getWeight();
      } else {
        sum++;
      }
    }
    return sum;
  }
  private double totalWeight(Iterable<Vector> data) {
    double sum = 0;
    for (Vector row : data) {
      if (row instanceof WeightedVector) {
        sum += ((WeightedVector) row).getWeight();
      } else {
        sum++;
      }
    }
    return sum;
  }
}
