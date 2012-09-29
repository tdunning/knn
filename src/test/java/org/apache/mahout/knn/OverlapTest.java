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
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.knn.generate.Normal;
import org.apache.mahout.knn.search.FastProjectionSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Measure speed and overlap for different search methods.
 */
public class OverlapTest {
    Logger log = LoggerFactory.getLogger(this.getClass());

    @Test
    public void testOverlap() {
        int dimension = 10;
        int numDataVectors = 1000000;
        int numQueries = 100;
        int depth = 2000;

        Matrix data = new DenseMatrix(numDataVectors, dimension);
        data.assign(new Normal());

        Matrix queries = new DenseMatrix(numQueries, dimension);
        queries.assign(new Normal());

        Vector queryNorms = queries.aggregateRows(new VectorFunction() {
            @Override
            public double apply(Vector f) {
                return f.getLengthSquared();
            }
        });

        Matrix scores = data.times(queries.transpose()).assign(Functions.mult(-2));
        for (MatrixSlice row : scores) {
            Vector v1 = data.viewRow(row.index());
            row.vector().assign(Functions.plus(v1.getLengthSquared()));
            row.vector().assign(queryNorms, Functions.PLUS);
        }
        log.warn("Reference scoring done");

        List<PriorityQueue<Score>> bestScores = Lists.newArrayList();
        for (int i = 0; i < numQueries; i++) {
            bestScores.add(new PriorityQueue<Score>(depth, Ordering.natural().reverse()));
        }

        for (MatrixSlice row : scores) {
            for (int i = 0; i < numQueries; i++) {
                final PriorityQueue<Score> q = bestScores.get(i);
                q.add(new Score(row.index(), row.vector().get(i)));
                if (q.size() > depth) {
                    q.poll();
                }
            }
        }

        List<Set<Integer>> reference = Lists.newArrayList();
        List<Double> radius = Lists.newArrayList();
        for (MatrixSlice query : queries) {
            Set<Integer> x = Sets.newHashSet();
            double furthest = -Double.MAX_VALUE;
            for (Score best : bestScores.get(query.index())) {
                x.add(best.index);
                furthest = Math.max(best.score, furthest);
            }
            reference.add(x);
            radius.add(furthest);
        }
        log.warn("Reference data stored");
        log.warn("Starting add with speedup of {}", numDataVectors / (dimension * 2.0 * depth * 4.0));

        Searcher sut = new FastProjectionSearch(dimension, new SquaredEuclideanDistanceMeasure(), dimension * 2, depth * 4);
        sut.addAll(data);
        log.warn("Added data with speedup of {}", numDataVectors / (dimension * 2.0 * depth * 4.0));

        long t0 = System.nanoTime();
        for (MatrixSlice query : queries) {
            List<WeightedVector> r = sut.search(query.vector(), depth);
            Set<Integer> x = Sets.newHashSet();
            for (WeightedVector vector : r) {
                x.add(vector.getIndex());
            }
            double overlap = Sets.intersection(reference.get(query.index()), x).size() / (double) depth;
            System.out.printf("%4.1f %10.3f\n", 100 * overlap, r.get(depth - 1).getWeight() / radius.get(query.index()));
        }
        long t1 = System.nanoTime();
        log.warn("Done query time = {}", String.format("%.4f", (t1 - t0) / 1e9 / numQueries));
    }

    private static class Score implements Comparable<Score> {
        int index;
        double score;

        public Score(int index, double score) {
            this.index = index;
            this.score = score;
        }

        @Override
        public int compareTo(Score other) {
            return Double.compare(score, other.score);
        }
    }
}
