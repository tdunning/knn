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
import java.util.TreeSet;
import java.util.Map;
import java.util.ArrayList;
import java.util.Comparator;

/**
 * Does approximate nearest neighbor dudes search by projecting the data.
 */
public class ProjectionSearch2 {
    private final List<TreeSet<WeightedVector>> vectors;
    
    private DistanceMeasure distance;
    private List<Vector> basis;

    public ProjectionSearch2(int d, DistanceMeasure distance) {
        this(d, distance, 1);
    }

    public ProjectionSearch2(int d, DistanceMeasure distance, int projections) {
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
     */
    public void add(Vector v) {
        // add to each projection separately
        Iterator<Vector> projections = basis.iterator();
        for (TreeSet<WeightedVector> s : vectors) {
            s.add(new WeightedVector(v, projections.next()));
        }
    }

    public List<Vector> search(final Vector query, int n, int searchSize) {
        
        Map<Vector, Double> allCustomers = Maps.newHashMap();
        for (WeightedVector element : vectors.get(0)) {
        	allCustomers.put(element.getVector(), 0.0);
        }
        
        Iterator<Vector> projections = basis.iterator();
        for (TreeSet<WeightedVector> v : vectors) {
            WeightedVector projectedQuery = new WeightedVector(query, projections.next());
            for (WeightedVector candidate : v) {
            	double dif = Math.abs(candidate.getWeight()-projectedQuery.getWeight());
            	
            	// dif += allCustomers.get(candidate.getVector());
            	// allCustomers.put(candidate.getVector(),dif);
            	
            	
            	if (dif > allCustomers.get(candidate.getVector())) {
            		allCustomers.put(candidate.getVector(),dif);
            	}
            	
            }
        }
        // System.out.printf("%d %d\n", candidates.size(), candidates.elementSet().size());

        // if searchSize * vectors.size() is small enough not to cause much memory pressure, this is probably
        // just as fast as a priority queue here.
        
        List custList = new ArrayList(allCustomers.entrySet());  
        
        Collections.sort(custList, new Comparator() {  
            public int compare(Object o1 , Object o2)  
            {  
                Map.Entry e1 = (Map.Entry)o1 ;  
                Map.Entry e2 = (Map.Entry)o2 ;  
                Double first = (Double)e1.getValue();  
                Double second = (Double)e2.getValue();  
                return first.compareTo(second);  
            }  
        });  
        
        List<Map.Entry> list1 = custList.subList(0, searchSize);
        Iterator<Map.Entry> list2 = list1.iterator();
        List<Vector> candidates = Lists.newArrayList();
        while (list2.hasNext()) {
        	Map.Entry entry = (Map.Entry)list2.next();
        	candidates.add((Vector)entry.getKey());
        } 
        Collections.sort(candidates, byQueryDistance(query));
        return candidates.subList(0, n);
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
