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

package org.apache.mahout.knn;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleDoubleFunction;

/**
 * A centroid is a weighted vector.  We have it delegate to the vector itself for lots of operations
 * to make it easy to use vector search classes and such.
 */
public class Centroid extends WeightedVector {
    private int key;

    public Centroid(Centroid original) {
        super(original.size(), original.getWeight());
        delegate = original.like();
        delegate.assign(original);
        key = original.getKey();
    }

    public Centroid(int key, Vector initialValue) {
        super(initialValue, 1);
        this.key = key;
        this.weight = 1;
    }

    public Centroid(int key, Vector initialValue, double weight) {
        this(key, initialValue);
        this.weight = weight;
    }

    public void update(final Centroid other) {
        update(other.delegate, other.weight);
    }

    public void update(Vector v) {
        update(v, 1);
    }

    public void update(Vector v, final double w) {
        final double totalWeight = weight + w;
        delegate.assign(v, new DoubleDoubleFunction() {
            @Override
            public double apply(double v, double v1) {
                return (weight * v + w * v1) / totalWeight;
            }
        });
        weight += w;
    }

    public int getKey() {
        return key;
    }

    public void setKey(int newKey) {
        this.key = newKey;

    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double newWeight) {
        this.weight = newWeight;
    }

    public void addWeight() {
        this.weight = this.weight + 1;
    }

    public String toString() {
        return new StringBuilder("key = ").append(String.valueOf(key)).append(", weight = ").append(String.valueOf(weight)).append(", delegate = ").append(delegate.toString()).toString();
    }
}
