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

package org.apache.mahout.knn;

/*
import com.google.common.collect.Lists;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.math.ConstantVector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.Sampler;
import org.apache.mahout.math.WeightedVector;

import java.util.List;

public class BruteSpeedCheck {
  private static final int VECTOR_DIMENSION = 250;
  private static final int REFERENCE_SIZE = 10000;
  private static final int QUERY_SIZE = 100;

  public static void main(String[] args) {
    Sampler<Vector> rand = new MultiNormal(new ConstantVector(1, VECTOR_DIMENSION));
    List<WeightedVector> referenceVectors = Lists.newArrayListWithExpectedSize(REFERENCE_SIZE);
    for (int i = 0; i < REFERENCE_SIZE; ++i) {
      referenceVectors.add(new WeightedVector(rand.sample(), 1, i));
    }
    System.out.printf("Generated reference matrix.\n");

    List<WeightedVector> queryVectors = Lists.newArrayListWithExpectedSize(QUERY_SIZE);
    for (int i = 0; i < QUERY_SIZE; ++i) {
      queryVectors.add(new WeightedVector(rand.sample(), 1, i));
    }
    System.out.printf("Generated query matrix.\n");

    for (int threads : new int[]{1, 2, 3, 4, 5, 6, 10, 20, 50}) {
      for (int block : new int[]{1, 10, 50}) {
        Brute search = new Brute(referenceVectors);
        long t0 = System.nanoTime();
        search.search(queryVectors, block, threads);
        long t1 = System.nanoTime();
        System.out.printf("%d\t%d\t%.2f\n", threads, block, (t1 - t0) / 1e9);
      }
    }
  }
}
*/
