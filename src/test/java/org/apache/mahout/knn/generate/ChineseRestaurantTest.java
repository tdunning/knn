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

package org.apache.mahout.knn.generate;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.collect.Ordering;
import org.junit.Assert;
import org.junit.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class ChineseRestaurantTest {
    @Test
    public void testDepth() {
        List<Integer> totals = Lists.newArrayList();
        for (int i = 0; i < 1000; i++) {
            ChineseRestaurant x = new ChineseRestaurant(10);
            Multiset<Integer> counts = HashMultiset.create();
            for (int j = 0; j < 100; j++) {
                counts.add(x.sample());
            }
            List<Integer> tmp = Lists.newArrayList();
            for (Integer k : counts.elementSet()) {
                tmp.add(counts.count(k));
            }
            Collections.sort(tmp, Ordering.natural().reverse());
            while (totals.size() < tmp.size()) {
                totals.add(0);
            }
            int j = 0;
            for (Integer k : tmp) {
                totals.set(j, totals.get(j) + k);
                j++;
            }
        }

        // these are empirically derived values, not principled ones
        assertEquals(20600.0, (double) totals.get(0), 1000);
        assertEquals(13200.0, (double) totals.get(1), 1000);
        assertEquals(9875.0, (double) totals.get(2), 200);
        assertEquals(1475.0, (double) totals.get(15), 50);
        assertEquals(880.0, (double) totals.get(20), 40);
    }
}
