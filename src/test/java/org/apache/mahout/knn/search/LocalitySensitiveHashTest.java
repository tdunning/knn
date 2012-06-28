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

import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.generate.Normal;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.knn.search.HashedVector;
import org.apache.mahout.knn.search.LocalitySensitiveHash;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.junit.Assert;
import org.junit.Test;

import java.util.BitSet;
import java.util.List;

public class LocalitySensitiveHashTest {

    @Test
    public void testNormal() {
        Matrix testData = new DenseMatrix(100000, 10);
        final Normal gen = new Normal();
        testData.assign(gen);

        Brute ref = new Brute(testData);
        final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();

        LocalitySensitiveHash cut = new LocalitySensitiveHash(10, distance, 10);
        cut.addAll(testData);

        cut.setSearchSize(200);
        cut.resetEvaluationCount();

        System.out.printf("speedup,q1,q2,q3\n");

        for (int i = 0; i < 12; i++) {
            double strategy = (i - 1.0) / 10.0;
            cut.setRaiseHashLimitStrategy(strategy);
            OnlineSummarizer t1 = evaluateStrategy(testData, ref, cut);
            int evals = cut.resetEvaluationCount();
            final double speedup = 10e6 / evals;
            System.out.printf("%.1f,%.2f,%.2f,%.2f\n", speedup, t1.getQuartile(1), t1.getQuartile(2), t1.getQuartile(3));
            Assert.assertTrue(t1.getQuartile(2) > 0.45);
            Assert.assertTrue(speedup > 4 || t1.getQuartile(2) > 0.9);
            Assert.assertTrue(speedup > 15 || t1.getQuartile(2) > 0.8);
        }
    }

    private OnlineSummarizer evaluateStrategy(Matrix testData, Brute ref, LocalitySensitiveHash cut) {
        OnlineSummarizer t1 = new OnlineSummarizer();

        for (int i = 0; i < 100; i++) {
            final Vector q = testData.viewRow(i);
            List<WeightedVector> v1 = cut.search(q, 150);
            BitSet b1 = new BitSet();
            for (WeightedVector v : v1) {
                b1.set(v.getIndex());
            }

            List<WeightedVector> v2 = ref.search(q, 100);
            BitSet b2 = new BitSet();
            for (WeightedVector v : v2) {
                b2.set(v.getIndex());
            }

            b1.and(b2);
            t1.add(b1.cardinality());
        }
        return t1;
    }

    @Test
    public void testDotCorrelation() {
        final Normal gen = new Normal();

        Matrix projection = new DenseMatrix(64, 10);
        projection.assign(gen);

        Vector query = new DenseVector(10);
        query.assign(gen);
        long qhash = HashedVector.computeHash64(query, projection);

        int count[] = new int[65];
        Vector v = new DenseVector(10);
        for (int i = 0; i <500000; i++) {
            v.assign(gen);
            long hash = HashedVector.computeHash64(v, projection);
            final int bitDot = Long.bitCount(qhash ^ hash);
            count[bitDot]++;
            if (count[bitDot] < 200) {
                System.out.printf("%d, %.3f\n", bitDot, v.dot(query) / Math.sqrt(v.getLengthSquared() * query.getLengthSquared()));
            }
        }
    }
}
