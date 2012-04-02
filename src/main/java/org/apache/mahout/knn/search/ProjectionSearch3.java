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
import com.google.common.collect.Sets;
import com.google.common.collect.Maps;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.Map;

/**
 * Does approximate nearest neighbor dudes search by projecting the data.
 */
public class ProjectionSearch3 extends Searcher {
    private final List<TreeSet<WeightedVector>> vectors;

    private DistanceMeasure distance;
    private List<Vector> basis;
    private int searchSize;

    public ProjectionSearch3(int d, DistanceMeasure distance, int projections, int searchSize) {
        this.searchSize = searchSize;
        Preconditions.checkArgument(projections > 0 && projections < 100, "Unreasonable value for number of projections");

        final DoubleFunction random = Functions.random();

        this.distance = distance;
        vectors = Lists.newArrayList();
        basis = Lists.newArrayList();

        // we want to create several projections.  Each is alike except for the
        // direction of the projection
        for (int i = 0; i < projections; i++) {
            // create a random vector to use for the basis of the projection
            final DenseVector projection = new DenseVector(d);
            projection.assign(random);
            projection.normalize();

            basis.add(projection);

            // the projection is implemented by a tree set where the ordering of vectors
            // is based on the dot product of the vector with the projection vector
            vectors.add(Sets.<WeightedVector>newTreeSet());
        }
    }

    /**
     * Adds a vector into the set of projections for later searching.
     * @param v  The vector to add.
     * @param index
     */
    public void add(Vector v, int index) {
        // add to each projection separately
        Iterator<Vector> projections = basis.iterator();
        for (TreeSet<WeightedVector> s : vectors) {
            s.add(new WeightedVector(v, projections.next(), index));
        }
    }

    public List<WeightedVector> search(final Vector query, int n) {
        // this is keyed by the underlying vector to make sure that comparisons
        // work right between different projections.  The value is a shallow copy of
        // the result vector so that we can set the weight to the actual distance from
        // the query
        Map<Vector, WeightedVector> distances = Maps.newHashMap();

        // for each projection
        Iterator<Vector> projections = basis.iterator();
        for (TreeSet<WeightedVector> v : vectors) {
            WeightedVector projectedQuery = WeightedVector.project(query, projections.next());

            // Collect nearby vectors
            List<WeightedVector> candidates = Lists.newArrayList();
            Iterables.addAll(candidates, Iterables.limit(v.tailSet(projectedQuery, true), searchSize));
            Iterables.addAll(candidates, Iterables.limit(v.headSet(projectedQuery, false).descendingSet(), searchSize));

            // find maximum projected distance in nearby values.
            // all unmentioned values will be at least that far away.
            // also collect a set of unmentioned values
            Set<Vector> unmentioned = Sets.newHashSet(distances.keySet());
            double maxDistance = 0;
            for (WeightedVector vector : candidates) {
                unmentioned.remove(vector.getVector());
                maxDistance = Math.max(maxDistance, vector.getWeight());
            }

            // all unmentioned vectors have to be put at least as far away as we can justify
            for (Vector vector : unmentioned) {
                WeightedVector x = distances.get(vector);
                if (maxDistance > x.getWeight()) {
                    x.setWeight(maxDistance);
                }
            }

            // and all candidates get a real test
            for (WeightedVector candidate : candidates) {
                WeightedVector x = distances.get(candidate);
                if (x == null) {
                    // have to copy here because we may mutate weights later on
                    distances.put(candidate.getVector(), new WeightedVector(candidate.getVector(), candidate.getWeight(), candidate.getIndex()));
                } else if (x.getWeight() < candidate.getWeight()) {
                    x.setWeight(candidate.getWeight());
                }
            }
        }

        // now collect the results and sort by actual distance
        // TODO It doesn't seem to make a great gob of sense to collect the max projected distance and then toss it away
        List<WeightedVector> r = Lists.newArrayList();
        for (Vector key : distances.keySet()) {
            WeightedVector x = distances.get(key);
            x.setWeight(distance.distance(query, key));
            r.add(x);
        }

        Collections.sort(r);
        return r.subList(0, n);
    }

    @Override
    public int size() {
        return vectors.get(0).size();
    }

    @Override
    public int getSearchSize() {
        return searchSize;
    }

    @Override
    public void setSearchSize(int searchSize) {
        this.searchSize = searchSize;
    }


}
