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
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.MultiNormal;

import java.util.List;

/**
 * A collection of miscellaneous utility functions for working with data to be clustered.
 * Includes methods for generating synthetic data and estimating distance cutoff.
 */
public class DataUtils {
  /**
   * Samples numDatapoints vectors of numDimensions cardinality centered around the vertices of a
   * numDimensions order hypercube. The distribution of points around these vertices is
   * multinormal with a radius of distributionRadius.
   * A hypercube of numDimensions has 2^numDimensions vertices. Keep this in mind when clustering
   * the data.
   *
   *
   * @param numDimensions number of dimensions of the vectors to be generated.
   * @param numDatapoints number of data points to be generated.
   * @param distributionRadius radius of the distribution around the hypercube vertices.
   * @return a pair of lists, whose first element is the sampled points and whose second element
   * is the list of hypercube vertices that are the means of each distribution.
   */
  public static Pair<List<Centroid>, List<Centroid>> sampleMultiNormalHypercube(
      int numDimensions, int numDatapoints, double distributionRadius) {
    int pow2N = 1 << numDimensions;
    // Construct data samplers centered on the corners of a unit hypercube.
    // Additionally, keep the means of the distributions that will be generated so we can compare
    // these to the ideal cluster centers.
    List<Centroid> mean = Lists.newArrayListWithCapacity(pow2N);
    List<MultiNormal> rowSamplers = Lists.newArrayList();
    for (int i = 0; i < pow2N; i++) {
      Vector v = new DenseVector(numDimensions);
      // Select each of the num
      int pow2J = 1 << (numDimensions - 1);
      for (int j = 0; j < numDimensions; ++j) {
        v.set(j, 1.0 / pow2J * (i & pow2J));
        pow2J >>= 1;
      }
      mean.add(new Centroid(i, v, 1));
      rowSamplers.add(new MultiNormal(distributionRadius, v));
    }

    // Sample the requested number of data points.
    List<Centroid> data = Lists.newArrayListWithCapacity(numDatapoints);
    for (int i = 0; i < numDatapoints; ++i) {
      data.add(new Centroid(i, rowSamplers.get(i % pow2N).sample(), 1));
    }
    return new Pair<List<Centroid>, List<Centroid>>(data, mean);
  }

  /**
   * Calls sampleMultinormalHypercube(numDimension, numDataPoints, 0.01).
   * @see DataUtils#sampleMultiNormalHypercube(int, int, double)
   */
  public static Pair<List<Centroid>, List<Centroid>> sampleMultiNormalHypercube(int numDimensions,
                                                                                int numDatapoints) {
    return sampleMultiNormalHypercube(numDimensions, numDatapoints, 0.01);
  }

  /**
   * Estimates the distance cutoff. In StreamingKMeans, the distance between two vectors divided
   * by this value is used as a probability threshold when deciding whether to form a new cluster
   * or not.
   * Small values (comparable to the minimum distance between two points) are preferred as they
   * guarantee with high likelihood that all but very close points are put in separate clusters
   * initially. The clusters themselves are actually collapsed periodically when their number goes
   * over the maximum number of clusters and the distanceCutoff is increased.
   * So, the returned value is only an initial estimate.
   * @param data
   * @param distanceMeasure
   * @param sampleLimit
   * @return the minimum distance between the first sampleLimit points
   * @see StreamingKMeans#clusterInternal(Iterable, boolean)
   */
  public static double estimateDistanceCutoff(Iterable<? extends Vector> data,
                                              DistanceMeasure distanceMeasure,
                                              int sampleLimit) {
    Iterable<? extends Vector> limitedData = Iterables.limit(data, sampleLimit);
    double minDistance = Double.POSITIVE_INFINITY;
    int i = 1;
    for (Vector u : limitedData) {
      for (Vector v : Iterables.skip(limitedData, i)) {
        double distance = distanceMeasure.distance(u, v);
        if (minDistance > distance) {
          minDistance = distance;
        }
      }
      ++i;
    }
    return minDistance;
  }

  /**
   * Calls estimateDistanceCutoff(data, EuclideanDistance, sampleLimit).
   * @see DataUtils#estimateDistanceCutoff(Iterable, org.apache.mahout.common.distance.DistanceMeasure, int)
   */
  public static double estimateDistanceCutoff(Iterable<? extends Vector> data, int sampleLimit) {
    return estimateDistanceCutoff(data, new EuclideanDistanceMeasure(), sampleLimit);
  }

  /**
   * Calls estimateDistanceCutoff(data, EuclideanDistanceMeasure, 100).
   * @see DataUtils#estimateDistanceCutoff(Iterable, org.apache.mahout.common.distance.DistanceMeasure, int)
   */
  public static double estimateDistanceCutoff(Iterable<? extends Vector> data) {
    return estimateDistanceCutoff(data, new EuclideanDistanceMeasure(), 100);
  }
}
