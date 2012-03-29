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
public class Centroid extends DelegatingVector {
    private double weight;
    private int key;
    private Vector delegate;

    public Centroid(Centroid original) {
        super(original.size());
        weight = original.getWeight();
        delegate = original.delegate.clone();
        key = original.getKey();
    }

    public Centroid(int key, Vector initialValue) {
        super(initialValue.size());
        this.key = key;
        this.delegate = initialValue.clone();
        this.weight = 1;
    }

    public void update(final Centroid other) {
        delegate.assign(other.delegate, new DoubleDoubleFunction() {
            double totalWeight = weight + other.weight;

            @Override
            public double apply(double v, double v1) {
                return (weight * v + other.weight * v1) / totalWeight;
            }
        });
        weight += other.weight;
    }
    
    public void replace(final Centroid other) {
        delegate.assign(other.delegate);
        weight = other.weight;
        
    }

    public int getKey() {
        return key;
    }

    public void setKey(int newKey) {
        this.key=newKey;
    
    }

    public double getWeight() {
        return weight;
    }
    
    public void setWeight(int newWeight) {
        this.weight=newWeight;
    }
    
    public void addWeight() {
        this.weight=this.weight+1;
    }
    
    public String toString() {
    	return new StringBuilder("key = ").append(String.valueOf(key)).append(", weight = ").append(String.valueOf(weight)).append(", delegate = ").append(delegate.toString()).toString();
    }
}