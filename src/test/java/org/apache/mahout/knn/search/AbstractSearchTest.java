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
import junit.framework.Assert;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.knn.Searcher;
import org.apache.mahout.knn.UpdatableSearcher;
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
import static org.junit.Assert.fail;

public abstract class AbstractSearchTest {
    protected static Matrix randomData() {
        Matrix data = new DenseMatrix(1000, 20);
        MultiNormal gen = new MultiNormal(20);
        for (MatrixSlice slice : data) {
            slice.vector().assign(gen.sample());
        }
        return data;
    }

    public abstract Iterable<MatrixSlice> testData();

    public abstract Searcher getSearch(int n);

    @Test
    public void testExactMatch() {
        Iterable<MatrixSlice> data = testData();

        final Iterable<MatrixSlice> batch1 = Iterables.limit(data, 300);
        List<WeightedVector> queries = subset(batch1, 100);
        Searcher s = getSearch(20);

        // adding the data in multiple batches triggers special code in some searchers
        s.addAll(batch1);
        assertEquals(300, s.size());

        Vector q = Iterables.get(data, 0).vector();
        List<WeightedVector> r = s.search(q, 2);
        assertEquals(0, r.get(0).minus(q).norm(1), 1e-8);

        final Iterable<MatrixSlice> batch2 = Iterables.limit(Iterables.skip(data, 300), 10);
        s.addAll(batch2);
        assertEquals(310, s.size());

        q = Iterables.get(data, 302).vector();
        r = s.search(q, 2);
        assertEquals(0, r.get(0).minus(q).norm(1), 1e-8);

        s.addAll(Iterables.skip(data, 310));
        assertEquals(Iterables.size(testData()), s.size());

        for (Vector query : queries) {
            r = s.search(query, 2);
            assertEquals("Distance has to be about zero", 0, r.get(0).getWeight(), 1e-6);
            assertEquals("Answer must be substantially the same as query", 0, r.get(0).minus(query).norm(1), 1e-8);
            assertTrue("Wrong answer must have non-zero distance", r.get(1).getWeight() > r.get(0).getWeight());
        }
    }

    @Test
    public void testNearMatch() {
        List<WeightedVector> queries = subset(testData(), 100);
        Searcher s = getSearch(20);
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

        Searcher s = getSearch(20);
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

    @Test
    public void testSmallSearch() {
        Matrix m = new DenseMatrix(8, 3);
        for (int i = 0; i < 8; i++) {
            m.viewRow(i).assign(new double[]{0.125 * (i & 4), i & 2, i & 1});
        }

        Searcher s = getSearch(3);
        s.addAll(m);
        for (MatrixSlice row : m) {
            final List<WeightedVector> r = s.search(row.vector(), 3);
            assertEquals(0, r.get(0).getWeight(), 1e-8);
            assertEquals(0, r.get(1).getWeight(), 0.5);
            assertEquals(0, r.get(2).getWeight(), 1);
        }
    }

    @Test
    public void testRemoval() {
        Searcher s = getSearch(20);
        s.addAll(testData());
        if (s instanceof UpdatableSearcher) {
            List<WeightedVector> x = subset(s, 2);
            int size0 = s.size();

            List<WeightedVector> r0 = s.search(x.get(0), 2);

            s.remove(x.get(0).getVector(), 1e-7);
            assertEquals(size0 - 1, s.size());

            List<WeightedVector> r = s.search(x.get(0), 1);
            assertTrue("Vector should be gone", r.get(0).getWeight() > 0);
            assertEquals("Previous second neighbor should be first", 0, r.get(0).minus(r0.get(1)).norm(1), 1e-8);

            s.remove(x.get(1).getVector(), 1e-7);
            assertEquals(size0 - 2, s.size());

            r = s.search(x.get(1), 1);
            assertTrue("Vector should be gone", r.get(0).getWeight() > 0);

            // vectors don't show up in iterator
            for (MatrixSlice v : s) {
                Assert.assertTrue(x.get(0).minus(v.vector()).norm(1) > 1e-8);
                Assert.assertTrue(x.get(1).minus(v.vector()).norm(1) > 1e-8);
            }
        } else {
            try {
                List<WeightedVector> x = subset(s, 2);
                s.remove(x.get(0), 1e-7);
                fail("Shouldn't be able to delete from " + s.getClass().getName());
            } catch (UnsupportedOperationException e) {
                // good enough that UOE is thrown
            }
        }
    }

    public List<WeightedVector> subset(Iterable<MatrixSlice> data, int n) {
        List<WeightedVector> r = Lists.newArrayList();
        Random gen = RandomUtils.getRandom();

        int i = 0;
        for (MatrixSlice row : data) {
            if (r.size() < n) {
                if (row.vector() instanceof WeightedVector) {
                    r.add((WeightedVector) row.vector());
                } else {
                    r.add(new WeightedVector(row.vector(), 0, i++));
                }
            } else {
                int k = gen.nextInt(row.index());
                if (k < r.size()) {
                    if (row.vector() instanceof WeightedVector) {
                        r.set(k, (WeightedVector) row.vector());
                    } else {
                        r.set(k, new WeightedVector(row.vector(), 0, i++));
                    }
                }
                i++;
            }
        }

        return r;
    }
}
