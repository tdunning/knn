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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class MultinomialTest {
  @Test(expected = IllegalArgumentException.class)
  public void testNoValues() {
    Multiset<String> emptySet = HashMultiset.create();
    new Multinomial<String>(emptySet, 10);
  }

  @Test
  public void testSingleton() {
    Multiset<String> oneThing = HashMultiset.create();
    oneThing.add("one");
    Multinomial<String> s = new Multinomial<String>(oneThing, 10);
    assertEquals("one", s.sample(0));
    assertEquals("one", s.sample(0.1));
    assertEquals("one", s.sample(1));
  }

  @Test
  public void testEvenSplit() {
    Multiset<String> stuff = HashMultiset.create();
    for (int i = 0; i < 5; i++) {
      stuff.add(i + "");
    }
    Multinomial<String> s = new Multinomial<String>(stuff, 5);
    final double EPSILON = 1e-15;

    Multiset<String> cnt = HashMultiset.create();
    for (int i = 0; i < 5; i++) {
      cnt.add(s.sample(i * 0.2));
      cnt.add(s.sample(i * 0.2 + EPSILON));
      cnt.add(s.sample((i + 1) * 0.2 - EPSILON));
    }

    assertEquals(5, cnt.elementSet().size());
    for (String v : cnt.elementSet()) {
      assertEquals(3, cnt.count(v));
    }
    assertTrue(cnt.contains(s.sample(1)));
    assertEquals(s.sample(1 - EPSILON), s.sample(1));
  }

  @Test
  public void testPrime() {
    List<String> data = Lists.newArrayList();
    for (int i = 0; i < 17; i++) {
      String s = "0";
      if ((i & 1) != 0) {
        s = "1";
      }
      if ((i & 2) != 0) {
        s = "2";
      }
      if ((i & 4) != 0) {
        s = "3";
      }
      if ((i & 8) != 0) {
        s = "4";
      }
      data.add(s);
    }

    Multiset<String> stuff = HashMultiset.create();

    for (String x : data) {
      stuff.add(x);
    }

    Multinomial<String> s0 = new Multinomial<String>(stuff, 5);
    Multinomial<String> s1 = new Multinomial<String>(stuff, 13);
    Multinomial<String> s2 = new Multinomial<String>(stuff, 103);
    final double EPSILON = 1e-15;

    Multiset<String> cnt = HashMultiset.create();
    for (int i = 0; i < 50; i++) {
      final double p0 = i * 0.02;
      final double p1 = (i + 1) * 0.02;
      cnt.add(s0.sample(p0));
      cnt.add(s0.sample(p0 + EPSILON));
      cnt.add(s0.sample(p1 - EPSILON));

      assertEquals(s0.sample(p0), s1.sample(p0));
      assertEquals(s0.sample(p0 + EPSILON), s1.sample(p0 + EPSILON));
      assertEquals(s0.sample(p1 - EPSILON), s1.sample(p1 - EPSILON));

      assertEquals(s0.sample(p0), s2.sample(p0));
      assertEquals(s0.sample(p0 + EPSILON), s2.sample(p0 + EPSILON));
      assertEquals(s0.sample(p1 - EPSILON), s2.sample(p1 - EPSILON));
    }

    assertEquals(s0.sample(0), s1.sample(0));
    assertEquals(s0.sample(0 + EPSILON), s1.sample(0 + EPSILON));
    assertEquals(s0.sample(1 - EPSILON), s1.sample(1 - EPSILON));
    assertEquals(s0.sample(1), s1.sample(1));

    assertEquals(s0.sample(0), s2.sample(0));
    assertEquals(s0.sample(0 + EPSILON), s2.sample(0 + EPSILON));
    assertEquals(s0.sample(1 - EPSILON), s2.sample(1 - EPSILON));
    assertEquals(s0.sample(1), s2.sample(1));

    assertEquals(5, cnt.elementSet().size());
    Map<String, Integer> ref = ImmutableMap.of("3", 36, "2", 18, "1", 8, "0", 18, "4", 70);
    for (String v : cnt.elementSet()) {
      assertEquals(ref.get(v).intValue(), cnt.count(v));
    }

    assertTrue(cnt.contains(s0.sample(1)));
    assertEquals(s0.sample(1 - EPSILON), s0.sample(1));
  }
}
