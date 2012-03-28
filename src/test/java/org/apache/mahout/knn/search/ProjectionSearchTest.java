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
import com.google.common.collect.Ordering;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ProjectionSearchTest {
    @Test
    public void testSearch() {
        final DoubleFunction random = Functions.random();
        final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();

        List<Vector> ref = Lists.newArrayList();
        for (int i = 0; i < 1000; i++) {
            Vector v = new DenseVector(20);
            v.assign(random);
            ref.add(v);
        }

        for (int d = 1; d < 10; d++) {
            ProjectionSearch ps = new ProjectionSearch(20, distance, d);

            for (Vector v : ref) {
                ps.add(v);
            }

            // exact search should always work very efficiently
            for (int i = 0; i < 500; i++) {
                final Vector query = new DenseVector(ref.get(i));
                List<Vector> r = ps.search(query, 3, 5);
                assertEquals(0, r.get(0).minus(query).norm(1) , 1e-7);
            }

            int errors = 0;
            for (int i = 0; i < 100; i++) {
                final Vector query = ref.get(0).like();
                query.assign(random);
                Ordering<Vector> queryOrder = new Ordering<Vector>() {
                    @Override
                    public int compare(Vector v1, Vector v2) {
                        return Double.compare(distance.distance(query, v1), distance.distance(query, v2));
                    }
                };

                List<Vector> r = ps.search(query, 3, 50);

                // do exhaustive search for reference
                Collections.sort(ref, queryOrder);

                // correct answer should be nearly the best most of the time
                if (ref.indexOf(r.get(0)) > (d > 3 ? 10 : 20)) {
                    errors++;
                }
            }
            int errorLimit = d > 2 ? 1 : 10;
            assertTrue(String.format("d = %d, errors = %d", d, errors), errors <= errorLimit);
            System.out.printf("%d\t%d\n", d, errors);
        }
    }
}
