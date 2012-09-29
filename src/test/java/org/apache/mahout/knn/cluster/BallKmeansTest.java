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

import org.apache.mahout.knn.Centroid;
import org.apache.mahout.knn.generate.MultiNormal;
import org.apache.mahout.math.ConstantVector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BallKmeansTest {

    private static final int K1 = 100;

    @Test
    public void testBasicClustering() {
        Matrix data = cubishTestData(1);

        BallKmeans r = new BallKmeans(6, data, 20);
        for (Centroid centroid : r) {
            for (int i = 0; i < 10; i++) {
                System.out.printf("%10.4f", centroid.get(i));
            }
            System.out.printf("\n");
        }
    }

    @Test
    public void testInitialization() {
        // start with super clusterable data
        Matrix data = cubishTestData(0.01);

        // just do initialization of ball k-means.  This should drop a point into each of the clusters
        BallKmeans r = new BallKmeans(6, data, 0);

        // put the centroids into a matrix
        Matrix x = new DenseMatrix(6, 5);
        int row = 0;
        for (Centroid c : r) {
            x.viewRow(row).assign(c.viewPart(0, 5));
            row++;
        }

        // verify that each column looks right.  Should contain zeros except for a single 6.
        final Vector columnNorms = x.aggregateColumns(new VectorFunction() {
            @Override
            public double apply(Vector f) {
                // return the sum of three discrepancy measures
                return Math.abs(f.minValue()) + Math.abs(f.maxValue() - 6) + Math.abs(f.norm(1) - 6);
            }
        });
        // verify all errors are nearly zero
        assertEquals(0, columnNorms.norm(1) / columnNorms.size(), 0.1);

        // verify that the centroids are a permutation of the original ones
        SingularValueDecomposition svd = new SingularValueDecomposition(x);
        Vector s = svd.getS().viewDiagonal().assign(Functions.div(6));
        assertEquals(5, s.getLengthSquared(), 0.05);
        assertEquals(5, s.norm(1), 0.05);
    }

    private Matrix cubishTestData(double radius) {
        Matrix data = new DenseMatrix(K1 + 5000, 10);
        int row = 0;

        MultiNormal g = new MultiNormal(radius, new ConstantVector(0, 10));
        for (int i = 0; i < K1; i++) {
            data.viewRow(row++).assign(g.sample());
        }

        for (int i = 0; i < 5; i++) {
            Vector m = new DenseVector(10);
            m.set(i, i == 0 ? 6 : 6);
            MultiNormal gx = new MultiNormal(radius, m);
            for (int j = 0; j < 1000; j++) {
                data.viewRow(row++).assign(gx.sample());
            }
        }
        return data;
    }
}
