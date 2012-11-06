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
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.WeightedThing;
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

  /**
   * Gets a searcher whose search size is n.
   * @param n
   * @return
   */
  public abstract Searcher getSearch(int n);

  @Test
  public void testExactMatch() {
    Iterable<MatrixSlice> data = testData();

    final Iterable<MatrixSlice> batch1 = Iterables.limit(data, 300);
    List<MatrixSlice> queries = Lists.newArrayList(Iterables.limit(batch1, 100));
    Searcher s = getSearch(20);

    // adding the data in multiple batches triggers special code in some searchers
    s.addAllMatrixSlices(batch1);
    assertEquals(300, s.size());

    Vector q = Iterables.get(data, 0).vector();
    List<WeightedThing<Vector>> r = s.search(q, 2);
    assertEquals(0, r.get(0).getValue().minus(q).norm(1), 1e-8);

    final Iterable<MatrixSlice> batch2 = Iterables.limit(Iterables.skip(data, 300), 10);
    s.addAllMatrixSlices(batch2);
    assertEquals(310, s.size());

    q = Iterables.get(data, 302).vector();
    r = s.search(q, 2);
    assertEquals(0, r.get(0).getValue().minus(q).norm(1), 1e-8);

    s.addAllMatrixSlices(Iterables.skip(data, 310));
    assertEquals(Iterables.size(testData()), s.size());

    for (MatrixSlice query : queries) {
      r = s.search(query.vector(), 2);
      assertEquals("Distance has to be about zero", 0, r.get(0).getWeight(), 1e-6);
      assertEquals("Answer must be substantially the same as query", 0,
          r.get(0).getValue().minus(query.vector()).norm(1), 1e-8);
      assertTrue("Wrong answer must have non-zero distance",
          r.get(1).getWeight() > r.get(0).getWeight());
    }
  }

  @Test
  public void testNearMatch() {
    List<MatrixSlice> queries = Lists.newArrayList(Iterables.limit(testData(), 100));
    Searcher s = getSearch(20);
    s.addAllMatrixSlicesAsWeightedVectors(testData());

    MultiNormal noise = new MultiNormal(0.01, new DenseVector(20));
    for (MatrixSlice slice : queries) {
      Vector query = slice.vector();
      final Vector epsilon = noise.sample();
      List<WeightedThing<Vector>> r0 = s.search(query, 2);
      query = query.plus(epsilon);
      List<WeightedThing<Vector>> r = s.search(query, 2);
      r = s.search(query, 2);
      assertEquals("Distance has to be small", epsilon.norm(2), r.get(0).getWeight(), 1e-5);
      assertEquals("Answer must be substantially the same as query", epsilon.norm(2),
          r.get(0).getValue().minus(query).norm(2), 1e-5);
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
    // s.setSearchSize(200);
    s.addAllMatrixSlices(testData());

    for (MatrixSlice query : queries) {
      List<WeightedThing<Vector>> r = s.search(query.vector(), 200);
      double x = 0;
      for (WeightedThing<Vector> thing : r) {
        assertTrue("Scores must be monotonic increasing", thing.getWeight() > x);
        x = thing.getWeight();
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
    s.addAllMatrixSlices(m);
    for (MatrixSlice row : m) {
      final List<WeightedThing<Vector>> r = s.search(row.vector(), 3);
      assertEquals(0, r.get(0).getWeight(), 1e-8);
      assertEquals(0, r.get(1).getWeight(), 0.5);
      assertEquals(0, r.get(2).getWeight(), 1);
    }
  }

  @Test
  public void testRemoval() {
    Searcher s = getSearch(20);
    s.addAllMatrixSlices(testData());
    if (s instanceof UpdatableSearcher) {
      List<Vector> x = Lists.newArrayList(Iterables.limit(s, 2));
      int size0 = s.size();

      List<WeightedThing<Vector>> r0 = s.search(x.get(0), 2);

      s.remove(x.get(0), 1e-7);
      assertEquals(size0 - 1, s.size());

      List<WeightedThing<Vector>> r = s.search(x.get(0), 1);
      assertTrue("Vector should be gone", r.get(0).getWeight() > 0);
      assertEquals("Previous second neighbor should be first", 0,
          r.get(0).getValue().minus(r0.get(1).getValue()).norm (1), 1e-8);

      s.remove(x.get(1), 1e-7);
      assertEquals(size0 - 2, s.size());

      r = s.search(x.get(1), 1);
      assertTrue("Vector should be gone", r.get(0).getWeight() > 0);

      // vectors don't show up in iterator
      for (Vector v : s) {
        Assert.assertTrue(x.get(0).minus(v).norm(1) > 1e-8);
        Assert.assertTrue(x.get(1).minus(v).norm(1) > 1e-8);
      }
    } else {
      try {
        List<Vector> x = Lists.newArrayList(Iterables.limit(s, 2));
        s.remove(x.get(0), 1e-7);
        fail("Shouldn't be able to delete from " + s.getClass().getName());
      } catch (UnsupportedOperationException e) {
        // good enough that UOE is thrown
      }
    }
  }

  /*
  public List<Vector> subset(Iterable<Vector> data, int n) {
    List<Vector> r = Lists.newArrayList();
    Random gen = RandomUtils.getRandom();

    int i = 0;
    for (Vector row : data) {
      if (r.size() < n) {
          r.add(row);
      } else {
        int k = gen.nextInt(row.getIndex() + 1);
        if (k < r.size()) {
            r.set(k, new WeightedVector(row.getVector(), 1, i++));
        }
        i++;
      }
    }

    return r;
  }
  */
}
