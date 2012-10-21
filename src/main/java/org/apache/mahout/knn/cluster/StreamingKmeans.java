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
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.Searcher;
import org.apache.mahout.knn.UpdatableSearcher;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.WeightedVector;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class StreamingKmeans {
    // this parameter should be greater than 1, but not too much greater.
    // keeping BETA small makes the characteristic size grow more slowly
    // and small values of characteristic size seem to make the clustering
    // a bit better.  Too small a value of BETA, however, means that we
    // have to collapse the set of centroids too often.
    private static final double BETA = 1.3;

    // this is the current value of the characteristic size.  Points
    // which are much closer than this to a centroid will stick to it
    // almost certainly.  Points further than this to any centroid will
    // form a new cluster.
    private double distanceCutoff;

    public Searcher cluster(final DistanceMeasure distance, Iterable<MatrixSlice> data, int maxClusters) {
        final int width = data.iterator().next().vector().size();
        return cluster(data, maxClusters, new SearchFactory() {
            @Override
            public UpdatableSearcher create() {
                return new ProjectionSearch(width, distance, 10, 20);
            }
        });
    }

    /**
     * Provides a way to plug in alternative searchers for use in the clustering operation.
     */
    public interface SearchFactory {
        UpdatableSearcher create();
    }

    public Searcher cluster(Iterable<MatrixSlice> data, int maxClusters, SearchFactory searchFactory) {
        // initialize scale
        distanceCutoff = estimateCutoff(data);

        // cluster the data
        return clusterInternal(data, maxClusters, 1, searchFactory);
    }

    public static double estimateCutoff(Iterable<MatrixSlice> data) {
        Iterable<MatrixSlice> top = Iterables.limit(data, 100);

        // first we need to have a reasonable value for what a "small" distance is
        // so we find the shortest distance between any of the first hundred data points
        double distanceCutoff = Double.POSITIVE_INFINITY;
        for (List<WeightedVector> distances : new Brute(top).search(top, 2)) {
            if (distances.size() > 1) {
                final double x = distances.get(1).getWeight();
                if (x != 0 && x < distanceCutoff) {
                    distanceCutoff = x;
                }
            }
        }
        return distanceCutoff;
    }

    private UpdatableSearcher clusterInternal(Iterable<MatrixSlice> data, int maxClusters, int depth, SearchFactory searchFactory) {

        // to cluster, we scan the data and either add each point to the nearest group or create a new group.
        // when we get too many groups, we need to increase the threshold and rescan our current groups
        Random rand = RandomUtils.getRandom();
        int n = 0;
        UpdatableSearcher centroids = searchFactory.create();
        centroids.add(Centroid.create(0, Iterables.get(data, 0).vector()), 0);

        for (MatrixSlice row : Iterables.skip(data, 1)) {
            // estimate distance d to closest centroid
            WeightedVector closest = centroids.search(row.vector(), 1).get(0);

            if (rand.nextDouble() < closest.getWeight() / distanceCutoff) {
                // add new centroid, note that the vector is copied because we may mutate it later
                centroids.add(Centroid.create(centroids.size(), row.vector()), centroids.size());
            } else {
                // merge against existing
                Centroid c = (Centroid) closest.getVector();
                centroids.remove(c, 1e-7);
                c.update(row.vector());
                centroids.add(c, c.getIndex());
            }

            if (depth < 2 && centroids.size() > maxClusters) {
                maxClusters = (int) Math.max(maxClusters, 10 * Math.log(n));
                // TODO does shuffling help?
                List<MatrixSlice> shuffled = Lists.newArrayList(centroids);
                Collections.shuffle(shuffled);
                centroids = clusterInternal(shuffled, maxClusters, depth + 1, searchFactory);

                // in the original algorithm, with distributions with sharp scale effects, the
                // distanceCutoff can grow to excessive size leading sub-clustering to collapse
                // the centroids set too much. This test prevents increase in distanceCutoff
                // the current value is doing fine at collapsing the clusters.
                if (centroids.size() > 0.2 * maxClusters) {
                    distanceCutoff *= BETA;
                }
            }
            n++;
        }
        return centroids;
    }
}
