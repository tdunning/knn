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

package org.apache.mahout.knn.search;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.LumpyData;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.Set;


public class FastProjectionSearchTest extends AbstractSearchTest {
  private static Matrix data;
  private static final int QUERIES = 20;
  private static final int SEARCH_SIZE = 300;
  private static final int MAX_DEPTH = 100;

  @BeforeClass
  public static void setUp() {
    data = randomData();
  }

  @Override
  public Iterable<MatrixSlice> testData() {
    return data;
  }

  @Test
  public void testEpsilon() {
    final int dataSize = 10000;
    final int querySize = 30;
    final DistanceMeasure metric = new EuclideanDistanceMeasure();

    // these determine the dimension for the test. Each scale is multiplied by each multiplier
    final List<Integer> scales = ImmutableList.of(10);
    final List<Integer> multipliers = ImmutableList.of(1, 2, 3, 5);

    for (Integer scale : scales) {
      for (Integer multiplier : multipliers) {
        int d = scale * multiplier;
        if (d == 1) {
          continue;
        }
        final Matrix data = new DenseMatrix(dataSize + querySize, d);
        final LumpyData clusters = new LumpyData(d, 0.05, 10);
        for (MatrixSlice row : data) {
          row.vector().assign(clusters.sample());
        }

        Matrix q = data.viewPart(0, querySize, 0, d);
        Matrix m = data.viewPart(querySize, dataSize, 0, d);

        BruteSearch brute = new BruteSearch(metric);
        brute.addAllMatrixSlices(m);
        FastProjectionSearch test = new FastProjectionSearch(d, metric, 20, 20);
        test.addAllMatrixSlices(m);

        int bigRatio = 0;
        double averageOverlap = 0;
        for (MatrixSlice qx : q) {
          final Vector query = qx.vector();
          final List<WeightedThing<WeightedVector>> r1 = brute.search(query, 20);
          WeightedVector v1 = r1.get(0).getValue();
          final List<WeightedThing<WeightedVector>> r2 = test.search(query, 30);
          WeightedVector v2 = r2.get(0).getValue();

          class StripWeight implements Function<WeightedThing<WeightedVector>, WeightedVector> {

            @Override
            public WeightedVector apply(WeightedThing<WeightedVector> input) {
              Preconditions.checkArgument(input != null);
              return input.getValue();
            }
          };

          for (WeightedVector v : Iterables.transform(r1, new StripWeight())) {
            for (WeightedVector w : Iterables.transform(r2, new StripWeight())) {
              if (v.equals(w))
                ++averageOverlap;
            }
          }
          if (v2.getWeight() / v1.getWeight() > 1.4) {
            bigRatio++;
          }
        }
        averageOverlap = averageOverlap / q.rowSize();

        Assert.assertTrue(bigRatio < 2);
        Assert.assertTrue(averageOverlap > 7);
      }
    }
  }

  @Override
  public UpdatableSearcher getSearch(int n) {
    return new FastProjectionSearch(n, new EuclideanDistanceMeasure(), 4, 20);
  }
}

