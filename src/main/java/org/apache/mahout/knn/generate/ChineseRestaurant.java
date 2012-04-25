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

import org.apache.mahout.math.list.DoubleArrayList;

import java.util.Random;

/**
 * Sample from an infinite dimensional multinomial whose parameters
 * are sampled from a Dirichlet process with parameter alpha.
 */
public class ChineseRestaurant implements Sampler<Integer> {
    private double alpha;
    private double weight = 0;
    private DoubleArrayList weights = new DoubleArrayList();

    private Random rand = new Random();

    public ChineseRestaurant(double alpha) {
        this.alpha = alpha;
    }

    public Integer sample() {
        double u = rand.nextDouble();
        final double pNew;
        pNew = alpha / (weight + alpha);

        if (u >= 1 - pNew) {
            weights.add(1);
            weight++;
            return weights.size() -1;
        } else {
            u = weight * u / (1 - pNew);
            for (int j = 0; j < weights.size(); j++) {
                if (u < weights.get(j)) {
                    weights.set(j, weights.get(j) + 1);
                    weight++;
                    return j;
                } else {
                    u -= weights.get(j);
                }
            }
            throw new RuntimeException("Can't happen!");
        }
    }

    public int size() {
        return weights.size();
    }
}
