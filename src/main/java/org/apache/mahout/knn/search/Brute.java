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

import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.WeightedEuclideanDistanceMeasure;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Search for nearest neighbors using complete search.
 */
// TODO make search return a weighted vector
public class Brute extends UpdatableSearcher {
    // matrix of vectors for comparison
    private final List<MatrixSlice> reference;

    private int limit = -1;
    private DistanceMeasure metric;

    public Brute(Vector weight) {
        this.reference = Lists.newArrayList();
        WeightedEuclideanDistanceMeasure m = new WeightedEuclideanDistanceMeasure();
        m.setWeights(weight);
        metric = m;
    }
    
    public Brute(DistanceMeasure metric) {
        this((Vector) null);
        this.metric = metric;
    }

    private Brute(Iterable<MatrixSlice> reference, Vector weight) {
        this.reference = Lists.newArrayList(reference);
        this.limit = Integer.MAX_VALUE;

        WeightedEuclideanDistanceMeasure m = new WeightedEuclideanDistanceMeasure();
        m.setWeights(weight);
        metric = m;
    }

    public Brute(Iterable<MatrixSlice> reference) {
        this(reference, null);
        metric = new EuclideanDistanceMeasure();
    }

    @Override
    public void add(Vector v, int index) {
        reference.add(new MatrixSlice(v, index));
    }

    /**
     * Searches for N neighbors of a single vector.
     *
     *
     * @param v The vector query to search for.
     * @param n The number of neighbors.
     * @return A list of neighbors ordered closest first.
     */
    public List<WeightedVector> search(Vector v, int n) {
        final List<WeightedVector> r = Lists.newArrayList(searchInternal(v, reference, n, new PriorityQueue<WeightedVector>(n, Ordering.natural().reverse())));
        Collections.sort(r);
        return r;
    }

    @Override
    public int size() {
        return reference.size();
    }

    @Override
    public int getSearchSize() {
        return limit;
    }

    @Override
    public void setSearchSize(int size) {
        this.limit = size;
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
    private PriorityQueue<WeightedVector> searchInternal(Vector query, Iterable<MatrixSlice> reference, int n, PriorityQueue<WeightedVector> q) {
        int rowNumber = 0;
        for (MatrixSlice row : reference) {
            if (limit > 0 && rowNumber > limit) {
                break;
            }
            double r = metric.distance(query, row.vector());
            
            // only insert if the result is better than the worst in the queue or the queue isn't full
            if (q.size() < n || q.peek().getWeight() > r) {
                q.add(new WeightedVector(row.vector(), r, row.index()));

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
    public List<List<WeightedVector>> search(Iterable<MatrixSlice> query, int n) {
        List<PriorityQueue<WeightedVector>> q = Lists.newArrayList();

        List<List<WeightedVector>> r = Lists.newArrayList();
        for (MatrixSlice row : query) {
            r.add(search(row.vector(), n));
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
    public List<List<WeightedVector>> search(Matrix query, final int n, int threads) {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        List<Callable<Object>> tasks = Lists.newArrayList();


        final List<List<WeightedVector>> results = Lists.newArrayList();
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

    @Override
    public Iterator<MatrixSlice> iterator() {
        return reference.iterator();
    }

    @Override
    public boolean remove(Vector query) {
        int rowNumber = 0;
        for (MatrixSlice row : reference) {
            if (limit > 0 && rowNumber > limit) {
                break;
            }
            double r = metric.distance(query, row.vector());

            // only insert if the result is better than the worst in the queue or the queue isn't full
            if (r < 1e-9) {
                reference.remove(rowNumber);
                return true;
            }
            rowNumber++;
        }
        return false;
    }
}
