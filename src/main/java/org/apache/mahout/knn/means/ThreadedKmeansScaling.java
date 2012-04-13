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

package org.apache.mahout.knn.means;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.generate.MultiNormal;
import org.apache.mahout.knn.generate.Sampler;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.knn.search.Searcher;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
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
        Sampler<Vector> gen = new MultiNormal(30);
        for (MatrixSlice row : data) {
            row.vector().assign(gen.sample());
        }


        Searcher y = new StreamingKmeans().cluster(new EuclideanDistanceMeasure(), data, 200);
        long t2 = System.currentTimeMillis();
        y = new StreamingKmeans().cluster(new EuclideanDistanceMeasure(), data, 200);
        long t3 = System.currentTimeMillis();
        for (int threads : new int[]{1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16}) {
            List<Iterable<MatrixSlice>> xdata = ThreadedKmeans.split(data, 3 * threads);
            long t0 = System.currentTimeMillis();
            Searcher x = new ThreadedKmeans().cluster(new EuclideanDistanceMeasure(), xdata, 200, threads);
            long t1 = System.currentTimeMillis();
            System.out.printf("%d\t%.2f\t%.2f\n", threads, (t3 - t2) / 1000.0 / rows * 1e6, (t1 - t0) / 1000.0 / rows * 1e6);
        }
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        int rows = 10000;
        if (args.length > 0) {
            rows = Integer.parseInt(args[0]);
        }
//        testScaling();
        testCluster(rows);
    }
}
