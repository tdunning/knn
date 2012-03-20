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

import org.apache.mahout.knn.generate.MultiNormal;
import org.apache.mahout.knn.generate.Sampler;
import org.apache.mahout.math.ConstantVector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;

/**
 * Tests brute force search speed.
 */
public class BruteSpeedCheck {
  public static void main(String[] args) {
    Sampler<Vector> rand = new MultiNormal(new ConstantVector(1, 250));
    Matrix ref = new DenseMatrix(10000, 250);
    for (MatrixSlice slice : ref) {
      slice.vector().assign(rand.sample());
    }
    System.out.printf("generated reference matrix\n");

    Matrix query = new DenseMatrix(100, 250);
    for (MatrixSlice slice : query) {
      slice.vector().assign(rand.sample());
    }
    System.out.printf("generated query matrix\n");

    for (int block : new int[]{100, 1, 5, 10, 20, 50, 100}) {
      Brute search = new Brute(ref, block);
      long t0 = System.nanoTime();
      search.search(query, 50);
      long t1 = System.nanoTime();
      System.out.printf("%d blocksize gives elapsed time = %.2f\n", block, (t1 - t0) / 1e9);
    }
  }
}
