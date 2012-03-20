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

import com.google.common.collect.Lists;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.util.List;
import java.util.PriorityQueue;

/**
 * Search for nearest neighbors using complete search.
 */
public class Brute {
  // matrix of vectors for comparison
  private final Matrix reference;

  // weight vectors for each coordinate
  private final Vector weight;

  // blocking factor
  private final int block;

  public Brute(int block, Matrix reference, Vector weight) {
    this.block = block;
    this.reference = reference;
    this.weight = weight;
  }

  public Brute(Matrix reference, Vector weight) {
    this(50, reference, weight);
  }

  public Brute(Matrix reference, int block) {
    this(block, reference, null);
  }

  public Brute(Matrix reference) {
    this(50, reference, null);
  }

  /**
   * Searches for N neighbors of a single vector.
   *
   *
   * @param v       The vector query to search for.
   * @param n       The number of neighbors.
   * @return        A list of neighbors ordered closest first.
   */
  public List<Result> search(Vector v, int n) {
    return Lists.reverse(Lists.newArrayList(searchInternal(v, reference, n, new PriorityQueue<Result>())));
  }

  /**
   * Scans a matrix one row at a time for neighbors of the query vector.
   * @param query        The query vector.
   * @param reference    The matrix to scan.
   * @param n            The number of results to return.
   * @param q            The queue to augment with results.
   * @return             The modified queue.
   */
  private PriorityQueue<Result> searchInternal(Vector query, Matrix reference, int n, PriorityQueue<Result> q) {
    for (MatrixSlice row : reference) {
      double r;
      if (weight != null) {
        r = query.minus(row.vector()).aggregate(weight, Functions.PLUS, new DoubleDoubleFunction() {
          @Override
          public double apply(double w, double diff) {
            return w * diff * diff;
          }
        });
      } else {
        r = query.minus(row.vector()).norm(2);
      }

      // only insert if the result is better than the worst in the queue or the queue isn't full
      if (q.size() < n || q.peek().score > r) {
        q.add(new Result(row.index(), r));

        while (q.size() > n) {
          q.poll();
        }
      }
    }
    return q;
  }

  /**
   * Searches with a matrix full of queries.
   * @param query    The queries to search for.
   * @param n        The number of results to return for each query.
   * @return         A list of result lists.
   */
  public List<List<Result>> search(Matrix query, int n) {
    List<PriorityQueue<Result>> q = Lists.newArrayList();

    final int queryRows = query.rowSize();
    final int referenceRows = reference.rowSize();

    // blocking is intended to allow better memory locality, but it seems to make little (5%) difference
    for (int i = 0; i < queryRows; i += block) {
      int queryBlockSize = Math.min(block, queryRows - i);
      final Matrix queryChunk = query.viewPart(i, queryBlockSize, 0, query.columnSize());

      for (int j = 0; j < referenceRows; j += block) {
        int referenceBlockSize = Math.min(block, referenceRows - j);
        final Matrix referenceBlock = reference.viewPart(j, referenceBlockSize, 0, reference.columnSize());

        for (MatrixSlice slice : queryChunk) {
          if (slice.index() + i >= q.size()) {
            q.add(new PriorityQueue<Result>());
          }
          searchInternal(slice.vector(), referenceBlock, n, q.get(slice.index() + i));
        }
      }
    }

    // convert the results to lists.
    List<List<Result>> r = Lists.newArrayList();
    for (PriorityQueue<Result> results : q) {
      r.add(Lists.reverse(Lists.newArrayList(results)));
    }
    return r;
  }

  public class Result implements Comparable<Result> {
    private int index;
    private double score;

    public Result(int index, double score) {
      this.index = index;
      this.score = score;
    }

    /**
     * Orders results descending by score and then ascending by id.
     * @param result  The result to compare to.
     * @return  An integer indicating the ordering between this and result.  
     */
    @Override
    public int compareTo(Result result) {
      int r = Double.compare(result.score, score);
      if (r == 0) {
        r = index - result.index;
      }
      return r;
    }
  }
}
