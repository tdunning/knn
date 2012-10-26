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
import com.sun.jdi.InvalidTypeException;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.WeightedThing;

import java.util.List;

/**
 * Describes how to search a bunch of weighted vectors.
 * In practice the weights of the vectors are not taken into account when
 * comparing the distances between vectors.
 *
 * When iterating through a Searcher, the WeightedVectors added to it are
 * returned.
 */
public abstract class Searcher implements Iterable<WeightedVector> {
  /**
   * Add a new WeightedVector to the Searcher that will be checked when getting
   * the nearest neighbors.
   */
  public abstract void add(WeightedVector v);

  /**
   * Returns the number of WeightedVectors being searched for nearest neighbors.
   */
  public abstract int size();

  /**
   * When querying the Searcher for the closest vectors, a list of
   * DistanceVector pairs is returned. The distance in a DistanceVectorPair is
   * the distance (calculated by some metric given in a concrete implementation
   * of the searcher) between the query vector and the vector in the pair.
   * The type of vector in the pair is always a WeightedVector.
   */
  public abstract List<WeightedThing<WeightedVector>> search(Vector query, int limit);

  /**
   * Adds all the data elements in the Searcher.
   *
   * @param data an iterable of WeightedVectors to add.
   */
  public void addAll(Iterable<WeightedVector> data) {
    for (WeightedVector v : data) {
        add(v);
    }
  }

  /**
   * Adds all the data elements in the Searcher.
   *
   * @param data an iterable of MatrixSlices to add.
   */
  public void addAllMatrixSlices(Iterable<MatrixSlice> data) {
    for (MatrixSlice slice : data) {
      add(new WeightedVector(slice.vector(), 1, slice.index()));
    }
  }

  public boolean remove(WeightedVector v, double epsilon) {
    throw new UnsupportedOperationException("Can't remove a vector from a "
        + this.getClass().getName());
  }

  public void clear() {
    throw new UnsupportedOperationException("Can't remove vectors from a "
        + this.getClass().getName());
  }
}
