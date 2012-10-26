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
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.random.WeightedThing;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * Does approximate nearest neighbor dudes search by projecting the data.
 */
public class FastProjectionSearch extends UpdatableSearcher implements
    Iterable<WeightedVector> {
  // the actual vectors
  private List<WeightedVector> data = Lists.newArrayList();
  private List<WeightedVector> pendingAdditions = Lists.newArrayList();

  // the projection basis
  private List<Vector> basis;

  // how we should measure distance
  private DistanceMeasure distanceMeasure;

  // how many candidates per projection to examine
  private int searchSize;

  // these store the projections of each vector against each basis vector
  private List<double[]> projections = Lists.newArrayList();
  // and these link back to the original data.
  private List<int[]> vectorIds = Lists.newArrayList();

  // dirty indicates that we have pending insertions
  private boolean dirty = false;
  // this keeps track of pending deletions that will get scrubbed away next time
  // we do a full index
  private int pendingRemovals = 0;

  public FastProjectionSearch(int d, DistanceMeasure distanceMeasure,
                              int projections, int searchSize) {
    this.searchSize = searchSize;
    Preconditions.checkArgument(projections > 0 && projections < 100,
        "Unreasonable value for number of projections");

    final DoubleFunction random = Functions.random();

    this.distanceMeasure = distanceMeasure;
    basis = Lists.newArrayList();

    // we want to create several projections.  Each is alike except for the
    // direction of the projection
    for (int i = 0; i < projections; i++) {
      // create a random vector to use for the basis of the projection
      final DenseVector projection = new DenseVector(d);
      projection.assign(random);
      projection.normalize();

      basis.add(projection);
    }
  }

  /**
   * Adds a vector into the set of projections for later searching.
   *
   * @param v     The weighted vector to add. It contains a weight (ignored
   *              for now) and an index which is used to track which vector
   *              is which.
   */
  public void add(WeightedVector v) {
    pendingAdditions.add(v);
  }

  @Override
  public boolean remove(WeightedVector vector, double epsilon) {
    boolean found = false;

    final double epsilon2 = epsilon * epsilon;
    Set<Integer> candidates = Sets.newHashSet();
    for (int i = 0; i < projections.size(); i++) {
      final double projection = basis.get(i).dot(vector);
      int r = Arrays.binarySearch(projections.get(i), projection);
      if (r < 0) {
        r = -(r + 1);
      }
      while (r >= 0 && projections.get(i)[r] >= projection - epsilon2) {
        r--;
      }
      while (r < data.size() && projections.get(i)[r] <= projection + epsilon2) {
        candidates.add(vectorIds.get(i)[r]);
        r++;
      }
    }

    for (Integer id : candidates) {
      if (id >= 0 && vector.getDistanceSquared(data.get(id)) < epsilon2) {
        // for all real matches, we just scrub away the id and take the storage
        // hit
        found = true;
        pendingRemovals++;

        // first set vector to something a vector that can't match
        Vector bogus = data.get(id).like();
        bogus.assign(Double.NaN);
        data.set(id, new WeightedVector(bogus, 0, -1));

        // then hide the tracks
        for (int[] idList : vectorIds) {
          for (int i = 0; i < idList.length; i++) {
            if (idList[i] == id) {
              idList[i] = -1;
            }
          }
        }
      }
    }

    if (pendingAdditions.size() > 0) {
      // copy just live vectors
      List<WeightedVector> newAdditions = Lists.newArrayList();
      for (WeightedVector v : pendingAdditions) {
        if (vector.getDistanceSquared(v) < epsilon2) {
          found = true;
        } else {
          newAdditions.add(v);
        }
      }
      pendingAdditions = newAdditions;
    }

    return found;
  }

  /**
   * Returns the number of vectors that we can search
   *
   * @return The number of vectors added to the search so far.
   */
  public int size() {
    return data.size() + pendingAdditions.size() - pendingRemovals;
  }


  public List<WeightedThing<WeightedVector>> search(final Vector query, int limit) {
    reindex();

    Multiset<Integer> candidateIds = HashMultiset.create();
    for (int i = 0; i < basis.size(); i++) {
      final double projection = basis.get(i).dot(query);
      int r = Arrays.binarySearch(projections.get(i), projection);
      if (r < 0) {
        r = -(r + 1);
      }
      int start = r - searchSize;
      int end = r + searchSize;
      if (start < 0) {
        start = 0;
      }
      if (end > data.size()) {
        end = data.size();
      }
      for (int j = start; j < end; j++) {
        candidateIds.add(vectorIds.get(i)[j]);
      }
    }

    // if searchSize * vectors.size() is small enough not to cause much memory
    // pressure, this is probably just as fast as a priority queue here.
    List<WeightedThing<WeightedVector>> top = Lists.newArrayList();
    for (int candidateId : candidateIds.elementSet()) {
      if (candidateId != -1) {
        final WeightedVector v = data.get(candidateId);
        top.add(new WeightedThing<WeightedVector>(v, distanceMeasure.distance(query, v)));
      }
    }

    Collections.sort(top);

    // may have to search the last few linearly
    if (pendingAdditions.size() > 0) {
      int lastIndex = top.size();
      if (lastIndex >= limit) {
       lastIndex = limit - 1;
       top = top.subList(0, lastIndex + 1);
      }
      double largestDistance = top.get(lastIndex).getWeight();

      for (WeightedVector v : pendingAdditions) {
        double distance = distanceMeasure.distance(query, v);
        if (distance < largestDistance) {
          top.add(new WeightedThing<WeightedVector>(v, distance));
        }
      }
      Collections.sort(top);
    }

    return top.subList(0, limit);
  }

  /**
   * If there are pending additions, we need to rebuild the indexes from
   * scratch.  Deletions don't require this, but we get rid of them in passing.
   */
  private void reindex() {
    if (dirty || pendingAdditions.size() > 0.05 * data.size()
        || pendingRemovals > 0.2 * data.size()) {
      data.addAll(pendingAdditions);
      pendingAdditions.clear();

      // clean up pending deletions by copying to a new list
      List<WeightedVector> newData = Lists.newArrayList();
      for (WeightedVector v : data) {
        if (!Double.isNaN(v.getQuick(0))) {
          newData.add(v);
        }
      }
      data = newData;
      pendingRemovals = 0;

      // build projections for all data
      vectorIds.clear();
      projections.clear();
      for (Vector u : basis) {
        List<WeightedThing<Integer>> tmp = Lists.newArrayList();
        int id = 0;
        for (WeightedVector vector : data) {
          tmp.add(new WeightedThing<Integer>(id++, u.dot(vector)));
        }
        Collections.sort(tmp);

        final int[] ids = new int[data.size()];
        vectorIds.add(ids);
        final double[] proj = new double[data.size()];
        projections.add(proj);
        int j = 0;
        for (WeightedThing<Integer> v : tmp) {
          ids[j] = v.getValue();
          proj[j] = v.getWeight();
          j++;
        }
      }
      dirty = false;
    }
  }

  /**
   * Marks a FastProjectionSearch as dirty if you modify any of the vectors
   * using iterators and such.
   */
  public void setDirty() {
    dirty = true;
  }

  public int getSearchSize() {
    return searchSize;
  }

  public void setSearchSize(int size) {
    searchSize = size;
  }

  @Override
  public Iterator<WeightedVector> iterator() {
    final int[] index = {0};
    return Iterators.concat(
        new AbstractIterator<WeightedVector>() {
          Iterator<WeightedVector> data =
              FastProjectionSearch.this.data.iterator();

          @Override
          protected WeightedVector computeNext() {
            if (!data.hasNext()) {
              return endOfData();
            } else {
              // get the next vector
              WeightedVector v = data.next();
              // but skip over deleted vectors
              while (Double.isNaN(v.get(0)) && data.hasNext()) {
                v = data.next();
              }
              // did we get a good one?
              if (!Double.isNaN(v.get(0))) {
                return v;
              } else {
                // no... ran out before we found a valid vector
                return endOfData();
              }
            }
          }
        },
        pendingAdditions.iterator());
  }

  @Override
  public void clear() {
    data.clear();
    vectorIds.clear();
    projections.clear();
    dirty = false;
  }
}
