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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.generate.MultiNormal;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public abstract class AbstractSearchTest {
    public abstract Iterable<MatrixSlice> testData();

    public abstract Searcher getSearch();

    @Test
    public void testExactMatch() {
        List<Vector> queries = subset(testData(), 100);
        Searcher s = getSearch();
        s.addAll(testData());
        assertEquals(Iterables.size(testData()), s.size());
        
        for (Vector query : queries) {
            List<WeightedVector> r = s.search(query, 2);
            assertEquals("Distance has to be about zero", 0, r.get(0).getWeight(), 1e-6);
            assertEquals("Answer must be substantially the same as query", 0, r.get(0).minus(query).norm(1), 1e-8);
            assertTrue("Wrong answer must have non-zero distance", r.get(1).getWeight() > r.get(0).getWeight());
        }
    }

    @Test
    public void testNearMatch() {
        List<Vector> queries = subset(testData(), 100);
        Searcher s = getSearch();
        s.addAll(testData());

        MultiNormal noise = new MultiNormal(0.01, new DenseVector(20));
        for (Vector query : queries) {
            final Vector epsilon = noise.sample();
            query = query.plus(epsilon);
            List<WeightedVector> r = s.search(query, 2);
            assertEquals("Distance has to be small", epsilon.norm(2), r.get(0).getWeight(), 1e-5);
            assertEquals("Answer must be substantially the same as query", epsilon.norm(2), r.get(0).minus(query).norm(2), 1e-5);
            assertTrue("Wrong answer must be further away", r.get(1).getWeight() > r.get(0).getWeight());
        }
    }
    
    @Test
    public void testOrdering() {
        Matrix queries = new DenseMatrix(100, 20);
        MultiNormal gen = new MultiNormal(20);
        for (int i = 0; i < 100; i++) {
            queries.viewRow(i).assign(gen.sample());
        }

        Searcher s = getSearch();
        s.setSearchSize(200);
        s.addAll(testData());

        for (MatrixSlice query : queries) {
            List<WeightedVector> r = s.search(query.vector(), Math.min(200, s.getSearchSize()));
            double x = 0;
            for (WeightedVector vector : r) {
                assertTrue("Scores must be monotonic increasing", vector.getWeight() > x);
                x = vector.getWeight();
            }
        }
    }

    public List<Vector> subset(Iterable<MatrixSlice> data, int n) {
        List<Vector> r = Lists.newArrayList();
        Random gen = RandomUtils.getRandom();

        for (MatrixSlice row : data) {
            if (r.size() < n) {
                r.add(row.vector());
            } else {
                int k = gen.nextInt(row.index());
                if (k < r.size()) {
                    r.set(k, row.vector());
                }
            }
        }

        return r;
    }
}
