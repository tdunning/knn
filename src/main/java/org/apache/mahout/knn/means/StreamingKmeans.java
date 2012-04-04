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
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.knn.Centroid;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixSlice;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class StreamingKmeans {
    private DistanceMeasure distance;
    private double distanceCutoff;

    public UpdatableSearcher cluster(DistanceMeasure distance, Iterable<MatrixSlice> data, int maxClusters) {
        // initialize scale
        distanceCutoff = estimateCutoff(data);
        this.distance = distance;

        // cluster the data
        UpdatableSearcher centroids = clusterInternal(data, maxClusters, 1);

        // how make a clean set of empty centroids to get ready for final pass through the data
        int width = data.iterator().next().vector().size();
        UpdatableSearcher r = new ProjectionSearch(width, distance, 4, 10);
        for (MatrixSlice centroid : centroids) {
            Centroid c = new Centroid(centroid.index(), new DenseVector(centroid.vector()));
            c.setWeight(0);
            r.add(c, c.getIndex());
        }

        // then make a final pass over the data
        for (MatrixSlice row : data) {
            WeightedVector closest = r.search(row.vector(), 1).get(0);

            // merge against existing
            Centroid c = (Centroid) closest.getVector();
            r.remove(c);
            c.update(row.vector());
            r.add(c, c.getIndex());
        }
        return r;
    }

    public double estimateCutoff(Iterable<MatrixSlice> data) {
        Iterable<MatrixSlice> top = Iterables.limit(data, 100);

        // first we need to have a reasonable value for what a "small" distance is
        // so we find the shortest distance between any of the first hundred data points
        distanceCutoff = Double.POSITIVE_INFINITY;
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

    private UpdatableSearcher clusterInternal(Iterable<MatrixSlice> data, int maxClusters, int depth) {
        int width = data.iterator().next().vector().size();
        UpdatableSearcher centroids = new ProjectionSearch(width, distance, 4, 10);

        // now we scan the data and either add each point to the nearest group or create a new group
        // when we get too many groups, then we need to increase the threshold and rescan our current groups
        Random rand = RandomUtils.getRandom();
        int n = 0;
        for (MatrixSlice row : data) {
            if (centroids.size() == 0) {
                // add first centroid on first vector
                centroids.add(new Centroid(centroids.size(), row.vector()), 0);
            } else {
                // estimate distance d to closest centroid
                WeightedVector closest = centroids.search(row.vector(), 1).get(0);

                if (rand.nextDouble() < closest.getWeight() / distanceCutoff) {
                    // add new centroid, note that the vector is copied because we may mutate it later
                    centroids.add(new Centroid(centroids.size(), new DenseVector(row.vector())), centroids.size());
                } else {
                    // merge against existing
                    Centroid c = (Centroid) closest.getVector();
                    centroids.remove(c);
                    c.update(row.vector());
                    centroids.add(c, c.getIndex());
                }
            }

            if (depth < 2 && centroids.size() > maxClusters) {
                maxClusters = (int) Math.max(maxClusters, 10 * Math.log(n));
                // TODO does shuffling help?
                List<MatrixSlice> shuffled = Lists.newArrayList(centroids);
                Collections.shuffle(shuffled);
                centroids = clusterInternal(shuffled, maxClusters, depth + 1);
                // for distributions with sharp scale effects, the distanceCutoff can grow to
                // excessive size leading sub-clustering to collapse the centroids set too much.
                // This test prevents that collapse from getting too severe.
                if (centroids.size() > 0.1 * maxClusters) {
                    distanceCutoff *= 1.5;
                }
            }
            n++;
        }
        return centroids;
    }
}
