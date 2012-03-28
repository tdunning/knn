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

import com.google.common.collect.Lists;
import org.apache.commons.collections.ListUtils;

import com.google.common.collect.Ordering;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import java.util.Collections;
import java.util.List;

public class ProjectionSearchTest {
    @Test
    public void testSearch() {
        final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();
        for (int d = 20; d < 21; d++) {
            ProjectionSearch ps = new ProjectionSearch(20, distance, d);
            List<Vector> ref = Lists.newArrayList();

            final DoubleFunction random = Functions.random();
            for (int i = 0; i < 40000; i++) {
                Vector v = new DenseVector(20);
                v.assign(random);
                ps.add(v);
                ref.add(v);
            }

            double sim = 0;
            int nSim = 0;
            double D1 = 0;
            double D2 = 0;
            double D3 = 0;
            int searchSize = 800;
            int returnSize = 100;
            List<Vector> randomNeighbor = Lists.newArrayList();
            randomNeighbor.addAll(ref.subList(0, returnSize));

            for (int i = 0; i < 100; i++) {
                // final Vector query = new DenseVector(ref.get(0));
                final Vector query = new DenseVector(20);
                query.assign(random);
                Ordering<Vector> queryOrder = new Ordering<Vector>() {
                    @Override
                    public int compare(Vector v1, Vector v2) {
                        return Double.compare(distance.distance(query, v1), distance.distance(query, v2));
                    }
                };


                List<Vector> r = ps.search(query, returnSize, searchSize);

                Collections.sort(ref, queryOrder);
                List<Vector> trueNeighbor = ref.subList(0, returnSize);
                List<Vector> proxyNeighbor = r.subList(0, returnSize);

                List<Vector> intersection1 = ListUtils.intersection(trueNeighbor, proxyNeighbor);
                List<Vector> union1 = ListUtils.sum(trueNeighbor, proxyNeighbor);
                // double jaccardSim = intersection1.size() / (double)union1.size();  
                // sim += jaccardSim;
                sim += intersection1.size() / (double) returnSize;
                nSim++;

                double d1 = 0;
                double d2 = 0;
                double d3 = 0;
                for (int j = 0; j < returnSize; j++) {
                    d1 += distance.distance(query, trueNeighbor.get(j));
                    d2 += distance.distance(query, proxyNeighbor.get(j));
                    d3 += distance.distance(query, randomNeighbor.get(j));
                    //System.out.print(distance.distance(query,trueNeighbor.get(j)));
                    //System.out.print(" ");
                    //System.out.println(distance.distance(query,randomNeighbor.get(j)));
                }
                d1 = d1 / returnSize;
                d2 = d2 / returnSize;
                d3 = d3 / returnSize;
                D1 += d1;
                D2 += d2;
                D3 += d3;

                /****
                 System.out.print(intersection1.size());
                 System.out.print(" ");
                 System.out.print(union1.size());
                 System.out.print(" ");
                 System.out.println(jaccardSim);
                 *****/
            }
            System.out.printf("d=%d ave_sim=%.2f trueNeighbor_dist=%.2f proxyNeighbor_dist=%.2f randomNeighbor_dist=%.2f \n", d, sim / nSim, D1 / nSim, D2 / nSim, D3 / nSim);
        }

    }
}
