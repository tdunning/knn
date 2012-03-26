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

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;

/**
 * Does approximate nearest neighbor dude search by projecting the data.
 */
public class ProjectionSearch {
    private final List<TreeSet<Vector>> vectors;
    
    private DistanceMeasure distance;

    public ProjectionSearch(int d, DistanceMeasure distance) {
        this(d, distance, 1);
    }

    public ProjectionSearch(int d, DistanceMeasure distance, int projections) {
        Preconditions.checkArgument(projections > 0 && projections < 100, "Unreasonable value for number of projections");

        final DoubleFunction random = Functions.random();

        this.distance = distance;
        vectors = Lists.newArrayList();

        for (int i = 0; i < projections; i++) {
            final DenseVector projection = new DenseVector(d);
            projection.assign(random);
            projection.normalize();

            TreeSet<Vector> s = Sets.newTreeSet(new Comparator<Vector>() {
                @Override
                public int compare(Vector v1, Vector v2) {
                    int r = Double.compare(v1.dot(projection), v2.dot(projection));
                    if (r == 0) {
                        return v1.hashCode() - v2.hashCode();
                    } else {
                        return r;
                    }
                }
            });
            vectors.add(s);
        }
    }

    public void add(Vector v) {
        for (TreeSet<Vector> s: vectors) {
            s.add(v);
        }
    }

    public List<Vector> search(final Vector query, int n, int searchSize) {
        List<Vector> top = Lists.newArrayList();
        for (TreeSet<Vector> v : vectors) {
            Iterables.addAll(top, Iterables.limit(v.tailSet(query, true), searchSize));
            Iterables.addAll(top, Iterables.limit(v.headSet(query, false).descendingSet(), searchSize));
        }

        // if searchSize * vectors.size() is small enough not to cause much memory pressure, this is probably
        // just as fast as a priority queue here.
        Collections.sort(top, byQueryDistance(query));
        return top.subList(0, n);
    }

    private Ordering<Vector> byQueryDistance(final Vector query) {
        return new Ordering<Vector>() {
            @Override
            public int compare(Vector v1, Vector v2) {
                int r = Double.compare(distance.distance(query, v1), distance.distance(query, v2));
                return r != 0 ? r : v1.hashCode() - v2.hashCode();
            }
        };
    }
}
