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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.knn.search.LocalitySensitiveHash;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.Searcher;
import org.apache.mahout.knn.UpdatableSearcher;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.ChineseRestaurant;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.Sampler;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.util.Collections;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;

/**
 * Demonstrate scaling performance.
 */
public class ThreadedKmeansScaling {
    private static final int QUERIES = 20;
    private static final int SEARCH_SIZE = 300;
    private static final int MAX_DEPTH = 100;

    private static final int THREADS = 2;
    private static final int SPLITS = 2;

    public void testClustering() throws ExecutionException, InterruptedException {

        // construct data samplers centered on the corners of a unit cube
        Matrix mean = new DenseMatrix(8, 3);
        List<MultiNormal> rowSamplers = Lists.newArrayList();
        for (int i = 0; i < 8; i++) {
            mean.viewRow(i).assign(new double[]{0.25 * (i & 4), 0.5 * (i & 2), i & 1});
            rowSamplers.add(new MultiNormal(0.01, mean.viewRow(i)));
        }

        // sample a bunch of data points
        Matrix data = new DenseMatrix(100000, 3);
        for (MatrixSlice row : data) {
            row.vector().assign(rowSamplers.get(row.index() % 8).sample());
        }

        // cluster the data
        final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();
        final int processors = Runtime.getRuntime().availableProcessors();
        System.out.printf("%d cores\n", processors);
        for (int i = 0; i < 4; i++) {
            for (int threads = 1; threads <= processors + 1; threads++) {
                for (int splits : new int[]{threads, 2 * threads}) {
                    clusterCheck(mean, "projection", data, new StreamingKmeans.SearchFactory() {
                        @Override
                        public UpdatableSearcher create() {
                            return new ProjectionSearch(3, distance, 4, 10);
                        }
                    }, threads, splits);
                    clusterCheck(mean, "lsh", data, new StreamingKmeans.SearchFactory() {
                        @Override
                        public UpdatableSearcher create() {
                            return new LocalitySensitiveHash(3, distance, 10);
                        }
                    }, threads, splits);

                }
            }
        }
    }

    private void clusterCheck(Matrix mean, String title, Matrix original, StreamingKmeans.SearchFactory searchFactory, int threads, int splits) throws ExecutionException, InterruptedException {
        List<Iterable<MatrixSlice>> data = Lists.newArrayList();
        int width = original.columnSize();
        int rowsPerSplit = (original.rowSize() + splits - 1) / splits;
        for (int i = 0; i < splits; i++) {
            final int rowOffset = i * rowsPerSplit;
            int rows = Math.min(original.rowSize() - rowOffset, rowsPerSplit);
            final Matrix split = original.viewPart(rowOffset, rows, 0, width).like();
            split.assign(original.viewPart(rowOffset, rows, 0, width));
            data.add(split);
        }

        long t0 = System.currentTimeMillis();
        Searcher r = new ThreadedKmeans().cluster(new EuclideanDistanceMeasure(), data, 1000, threads, searchFactory);
        long t1 = System.currentTimeMillis();

        // and verify that each corner of the cube has a centroid very nearby
        for (MatrixSlice row : mean) {
            WeightedVector v = r.search(row.vector(), 1).get(0);
        }
        System.out.printf("%s\t%d\t%d\t%.1f\n",
                title, threads, splits, (t1 - t0) / 1000.0 / original.rowSize() * 1e6);
    }

    private double totalWeight(Iterable<MatrixSlice> data) {
        double sum = 0;
        for (MatrixSlice row : data) {
            if (row.vector() instanceof WeightedVector) {
                sum += ((WeightedVector) row.vector()).getWeight();
            } else {
                sum++;
            }
        }
        return sum;
    }

    public static void testScaling() {
        DistanceMeasure m = new EuclideanDistanceMeasure();

        for (int d : new int[]{5, 10, 20, 50, 100, 200, 500}) {
            MultiNormal gen = new MultiNormal(d);

            final DenseVector projection = new DenseVector(d);
            projection.assign(gen.sample());
            projection.normalize();

            Matrix data = new DenseMatrix(10000, d);
            TreeSet<WeightedVector> tmp = Sets.newTreeSet();
            for (MatrixSlice row : data) {
                row.vector().assign(gen.sample());
                tmp.add(WeightedVector.project(row.vector(), projection, row.index()));
            }

            Searcher ref = new Brute(data);

            double correct = 0;
            double depthSum = 0;
            double[] cnt = new double[MAX_DEPTH];
            OnlineSummarizer distanceRatio = new OnlineSummarizer();
            for (int i = 0; i < QUERIES; i++) {
                Vector query = new DenseVector(d);
                query.assign(gen.sample());

                List<WeightedVector> nearest = ref.search(query, MAX_DEPTH);

                WeightedVector qp = WeightedVector.project(query, projection);
                List<WeightedVector> r = Lists.newArrayList();
                for (WeightedVector v : Iterables.limit(tmp.tailSet(qp, false), SEARCH_SIZE)) {
                    final WeightedVector projectedVector = new WeightedVector(v.getVector(), m.distance(query, v), v.getIndex());
                    r.add(projectedVector);
                }
                for (WeightedVector v : Iterables.limit(tmp.headSet(qp, false).descendingSet(), SEARCH_SIZE)) {
                    r.add(new WeightedVector(v.getVector(), m.distance(query, v), v.getIndex()));
                }
                Collections.sort(r);

                distanceRatio.add(r.get(0).getWeight() / nearest.get(0).getWeight());

                if (nearest.get(0).getIndex() == r.get(0).getIndex()) {
                    correct++;
                }
                int depth = 0;
                for (WeightedVector vector : nearest) {
                    if (vector.getIndex() == r.get(0).getIndex()) {
                        depthSum += depth;
                        cnt[depth]++;
                        break;
                    }
                    depth++;
                }
            }
            System.out.printf("%d\t%.2f\t%.2f", d, correct / QUERIES, depthSum / QUERIES);
            System.out.printf("\t%.2f\t%.2f\t%.2f", distanceRatio.getQuartile(1), distanceRatio.getQuartile(2), distanceRatio.getQuartile(3));
            for (int i = 0; i < 10; i++) {
                System.out.printf("\t%.2f", cnt[i] / QUERIES);
            }
            System.out.printf("\n");
        }

    }

    public static void testCluster(int rows) throws ExecutionException, InterruptedException {
        Matrix data = new DenseMatrix(rows, 30);
        PointSampler gen = new PointSampler(1);
        for (MatrixSlice row : data) {
            final Vector sample = gen.sample();
            row.vector().assign(sample);
        }
        System.out.printf("%d\n", gen.size());

        Matrix test = new DenseMatrix(1000, 30);
        for (MatrixSlice row : test) {
            row.vector().assign(gen.sample());
        }

        System.out.printf("%d\n", gen.size());

        final int width = 30;
        final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();

        final StreamingKmeans.SearchFactory psFactory = new StreamingKmeans.SearchFactory() {
            @Override
            public UpdatableSearcher create() {
                return new ProjectionSearch(width, distance, 4, 10);
            }
        };

        final StreamingKmeans.SearchFactory bruteFactory = new StreamingKmeans.SearchFactory() {
            @Override
            public UpdatableSearcher create() {
                return new ProjectionSearch(width, distance, 10, 20);
            }
        };

        long t4 = System.currentTimeMillis();
        Searcher y = new StreamingKmeans().cluster(data, 200, psFactory);
        long t5 = System.currentTimeMillis();
        double[] refRMSE = rmse(test, y);

        Searcher z1 = new ProjectionSearch(30, distance, 4, 10);
        Searcher z2 = new ProjectionSearch(30, distance, 10, 20);
        for (MultiNormal cluster : gen.gen) {
            z1.add(cluster.getMean(), -1);
            z2.add(cluster.getMean(), -1);
        }

        double[] z1RMSE = rmse(test, z1);
        double[] z2RMSE = rmse(test, z2);
        System.out.printf("%.2f\t%.2f\t%.2f\t%.2f\n", z1RMSE[0], z1RMSE[1], z2RMSE[0], z2RMSE[1]);


        System.out.printf("%.2f\n", (t5 - t4) / 1000.0 / rows * 1e6);


        for (int threads : new int[]{1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16}) {
            List<Iterable<MatrixSlice>> xdata = ThreadedKmeans.split(data, 2 * threads);
            long t0 = System.nanoTime();

            Searcher x = new ThreadedKmeans().cluster(distance, xdata, 200, threads, psFactory);
            long t1 = System.nanoTime();
            double[] psRMSE = rmse(test, x);

            long t2 = System.nanoTime();
            x = new ThreadedKmeans().cluster(distance, xdata, 200, threads, bruteFactory);
            long t3 = System.nanoTime();
            double[] bruteRMSE = rmse(test, x);

            System.out.printf("%d\t%.2f\t%.2f\t", threads, (t3 - t2) / 1e9 / rows * 1e6, (t1 - t0) / 1e9 / rows * 1e6);
//            System.out.printf("%.2f\t%.2f\n", refRMSE[0], refRMSE[1]);
//                    , psRMSE[0], psRMSE[1]);
            System.out.printf("%.2f\t%.2f\t%.2f\t%.2f\t%2f\t%.2f\n", refRMSE[0], refRMSE[1], psRMSE[0], psRMSE[1], bruteRMSE[0], bruteRMSE[1]);
        }


    }

    private static class PointSampler implements Sampler<Vector> {
        List<MultiNormal> gen = Lists.newArrayList();
        Sampler<Vector> means = new MultiNormal(30);
        ChineseRestaurant clusterId = new ChineseRestaurant(5);
        private double radius;

        private PointSampler(double radius) {
            this.radius = radius;
        }

        public Vector sample() {
            int id = clusterId.sample();
            while (gen.size() <= id) {
                gen.add(new MultiNormal(radius, means.sample()));
            }
            return gen.get(id).sample();
        }

        public int size() {
            return clusterId.size();
        }
    }

    private static double[] rmse(Matrix data, Searcher s) {
        Brute ref = new Brute(new EuclideanDistanceMeasure());
        ref.addAll(s);

        double sum1 = 0;
        double sum2 = 0;
        int rows = 0;
        for (MatrixSlice row : data) {
            WeightedVector v = ref.search(row.vector(), 1).get(0);
            sum1 += v.getWeight() * v.getWeight();

            v = s.search(row.vector(), 1).get(0);
            sum2 += v.getWeight() * v.getWeight();
            rows++;
        }
        sum1 = Math.sqrt(sum1 / rows / data.columnSize());
        sum2 = Math.sqrt(sum2 / rows / data.columnSize());
        return new double[]{sum1, sum2};
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        new ThreadedKmeansScaling().testClustering();

        int rows = 10000;
        if (args.length > 0) {
            rows = Integer.parseInt(args[0]);
        }
//        testScaling();
        testCluster(rows);
    }
}
