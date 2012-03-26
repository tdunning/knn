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
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Search for nearest neighbors using complete search.
 */
public class Brute {
    // matrix of vectors for comparison
    private final VectorIterable reference;

    // weight vectors for each coordinate
    private final Vector weight;

    // blocking factor
    private int limit;

    public Brute(VectorIterable reference, Vector weight) {
        this.reference = reference;
        this.weight = weight;
        this.limit = Integer.MAX_VALUE;
    }

    public Brute(VectorIterable reference) {
        this(reference, null);
    }

    /**
     * Searches for N neighbors of a single vector.
     *
     * @param v The vector query to search for.
     * @param n The number of neighbors.
     * @return A list of neighbors ordered closest first.
     */
    public List<Result> search(Vector v, int n) {
        return Lists.reverse(Lists.newArrayList(searchInternal(v, reference, n, new PriorityQueue<Result>())));
    }

    /**
     * Scans a matrix one row at a time for neighbors of the query vector.
     *
     * @param query     The query vector.
     * @param reference The matrix to scan.
     * @param n         The number of results to return.
     * @param q         The queue to augment with results.
     * @return The modified queue.
     */
    private PriorityQueue<Result> searchInternal(Vector query, VectorIterable reference, int n, PriorityQueue<Result> q) {
        int rowNumber = 0;
        for (MatrixSlice row : reference) {
            if (rowNumber > limit) {
                break;
            }
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
            rowNumber++;
        }
        return q;
    }

    /**
     * Searches with a matrix full of queries.
     *
     * @param query The queries to search for.
     * @param n     The number of results to return for each query.
     * @return A list of result lists.
     */
    public List<List<Result>> search(VectorIterable query, int n) {
        List<PriorityQueue<Result>> q = Lists.newArrayList();

        List<List<Result>> r = Lists.newArrayList();
        for (MatrixSlice row : query) {
            r.add(Lists.reverse(Lists.newArrayList(searchInternal(row.vector(), reference, n, q.get(row.index())))));
        }
        return r;
    }

    /**
     * Searches with a matrix full of queries in a threaded fashion.
     * @param query     The query.
     * @param n         Number of results to return.
     * @param threads   Number of threads to use in searching.
     * @return
     */
    public List<List<Result>> search(Matrix query, final int n, int threads) {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        List<Callable<Object>> tasks = Lists.newArrayList();


        final List<List<Result>> results = Lists.newArrayList();
        for (final MatrixSlice row : query) {
            results.add(null);
            tasks.add(new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                    results.set(row.index(), Brute.this.search(row.vector(), n));
                    return null;
                }
            });
        }

        try {
            es.invokeAll(tasks);
            es.shutdown();
        } catch (InterruptedException e) {
            throw new RuntimeException("Impossible error");
        }

        return results;
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
         *
         * @param result The result to compare to.
         * @return An integer indicating the ordering between this and result.
         */
        @Override
        public int compareTo(Result result) {
            int r = Double.compare(result.score, score);
            if (r == 0) {
                r = index - result.index;
            }
            return r;
        }

        public int getIndex() {
            return index;
        }

        public double getScore() {
            return score;
        }
    }
}
