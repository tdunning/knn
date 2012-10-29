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
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.WeightedThing;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Search for nearest neighbors using a complete search (i.e. looping through
 * the references and comparing each vector to the query).
 */
public class BruteSearch extends UpdatableSearcher {
  /**
   * The list of reference vectors.
   */
  private List<Vector> referenceVectors;

  public BruteSearch(DistanceMeasure distanceMeasure) {
    super(distanceMeasure);
    referenceVectors = Lists.newArrayList();
  }

  @Override
  public void add(Vector v) {
    referenceVectors.add(v);
  }

  @Override
  public int size() {
    return referenceVectors.size();
  }

  /**
   * Scans the list of reference vectors one at a time for @limit neighbors of
   * the query vector.
   * The weights of the WeightedVectors are not taken into account.
   *
   * @param query     The query vector.
   * @param limit The number of results to returned; must be at least 1.
   * @return A list of the closest @limit neighbors for the given query.
   */
  public List<WeightedThing<Vector>> search(Vector query, int limit) {
    Preconditions.checkArgument(limit > 0);
    limit = Math.min(limit, referenceVectors.size());
    // A priority queue of the best @limit elements, ordered from worst to best so that the worst
    // element is always on top and can easily be removed.
    PriorityQueue<WeightedThing<Integer>> bestNeighbors = new
        PriorityQueue<WeightedThing<Integer>>(limit, Ordering.natural().reverse());
    // The reulting list of weighted WeightedVectors (the weight is the distance from the query).
    List<WeightedThing<Vector>> results =
        Lists.newArrayListWithCapacity(limit);
    int rowNumber = 0;
    for (Vector row : referenceVectors) {
      double distance = distanceMeasure.distance(query, row);
      // Only add a new neighbor if the result is better than the worst element
      // in the queue or the queue isn't full.
      if (bestNeighbors.size() < limit || bestNeighbors.peek().getWeight() > distance) {
        bestNeighbors.add(new WeightedThing<Integer>(rowNumber, distance));
        if (bestNeighbors.size() > limit) {
          bestNeighbors.poll();
        } else {
          // Increase the size of the results list by 1 so we can add elements in the reverse
          // order from the queue.
          results.add(null);
        }
      }
      ++rowNumber;
    }
    for (int i = limit - 1; i >= 0; --i) {
      WeightedThing<Integer> neighbor = bestNeighbors.poll();
      results.set(i, new WeightedThing<Vector>(
          referenceVectors.get(neighbor.getValue()), neighbor.getWeight()));
    }
    return results;
  }

  /**
   * Searches with a list full of queries in a threaded fashion.
   *
   * @param queries The queries to search for.
   * @param limit The number of results to return.
   * @param numThreads   Number of threads to use in searching.
   * @return A list of result lists.
   */
  public List<List<WeightedThing<Vector>>> search(Iterable<WeightedVector> queries,
                                                  final int limit, int numThreads) {
    ExecutorService es = Executors.newFixedThreadPool(numThreads);
    List<Callable<Object>> tasks = Lists.newArrayList();

    final List<List<WeightedThing<Vector>>> results = Lists.newArrayList();
    int i = 0;
    for (final Vector query : queries) {
      results.add(null);
      final int index = i++;
      tasks.add(new Callable<Object>() {
        @Override
        public Object call() throws Exception {
          results.set(index, BruteSearch.this.search(query, limit));
          return null;
        }
      });
    }

    try {
      es.invokeAll(tasks);
      es.shutdown();
    } catch (InterruptedException e) {
      throw new RuntimeException("Impossible error");
    }

    return results;
  }

  @Override
  public Iterator<Vector> iterator() {
    return referenceVectors.iterator();
  }

  @Override
  public boolean remove(Vector query, double epsilon) {
    int rowNumber = 0;
    for (Vector row : referenceVectors) {
      double distance = distanceMeasure.distance(query, row);
      if (distance < epsilon) {
        referenceVectors.remove(rowNumber);
        return true;
      }
      rowNumber++;
    }
    return false;
  }

  @Override
  public void clear() {
    referenceVectors.clear();
  }
}
