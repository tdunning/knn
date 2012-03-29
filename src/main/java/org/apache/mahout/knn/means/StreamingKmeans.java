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

import com.google.common.collect.AbstractIterator;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.knn.Brute;
import org.apache.mahout.knn.Centroid;
import org.apache.mahout.knn.VectorIterableView;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.VectorIterable;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class StreamingKmeans {

    private final int width;
    private DistanceMeasure distance;
    private double beta;

    public StreamingKmeans(VectorIterable data, DistanceMeasure distance, int maxClusters) {
        this.distance = distance;
        VectorIterable top = new VectorIterableView(data, 0, 100);

        // first we need to have a reasonable value for what a "small" distance is
        beta = Double.POSITIVE_INFINITY;
        for (List<Brute.Result> distances : new Brute(top).search(top, 1)) {
            final double x = distances.get(0).getScore();
            if (x != 0 && x < beta) {
                beta = x;
            }
        }

        width = top.iterator().next().vector().size();
        cluster(data, maxClusters);
    }

    private <T extends Iterable<MatrixSlice>> ProjectionSearch cluster(T data, int maxClusters) {
        ProjectionSearch centroids = new ProjectionSearch(width, distance, 4);

        // now we scan the data and either add each point to the nearest group or create a new group
        // when we get too many groups, then we need to increase the threshold and rescan our current groups
        Random rand = RandomUtils.getRandom();
        for (MatrixSlice row : data) {
            if (centroids.size() == 0) {
                centroids.add(row.vector());
            } else {
                // estimate distance d to closest centroid
                WeightedVector closest = centroids.search(row.vector(), 10, 1).get(0);

                if (rand.nextDouble() < closest.getWeight() / beta) {
                    Centroid c = (Centroid) closest.getVector();
                    c.update(row.vector());
                } else {
                    centroids.add(new Centroid(centroids.size(), row.vector()));
                }
            }
            
            while (centroids.size() > maxClusters) {
                beta *= 1.5;
                final Collection<WeightedVector> v = centroids.getVectors();
                centroids = cluster(new Iterable<MatrixSlice>() {
                    @Override
                    public Iterator<MatrixSlice> iterator() {
                        return new AbstractIterator<MatrixSlice>() {
                            int index = 0;
                            Iterator<WeightedVector> data = v.iterator();

                            @Override
                            protected MatrixSlice computeNext() {
                                return new MatrixSlice(data.next(), index++);
                            }
                        };
                    }
                }, maxClusters);
            }
        }
        return centroids;
    }
}
