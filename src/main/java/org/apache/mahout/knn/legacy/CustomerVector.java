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

package org.apache.mahout.knn.legacy;

import org.apache.mahout.knn.DelegatingVector;
import org.apache.mahout.math.Vector;

/**
 * A CustomerVector is a DelegatingVector decorated with customer specific values.  
 */
public class CustomerVector extends DelegatingVector {
	private double performance=-1;
    private double euclideanDist=-1;
    private int clusterKey = -1;
    private String customerKey;

    public CustomerVector(final String customerKey, final Vector dataPoints, 
    		final int clusterKey, final double performance) {
        super(dataPoints);
        this.customerKey = customerKey;
        this.clusterKey=clusterKey;
        this.performance=performance;
    }

    public CustomerVector(final String customerKey, final Vector dataPoints) {
        super(dataPoints);
        this.customerKey = customerKey;
    }

    public CustomerVector(final String customerKey, final int size) {
    	super(size);
        this.customerKey = customerKey;
    }

    public CustomerVector(final int size) {
    	super(size);
    }
    
    public double getPerformance() {
		return performance;
	}

	public CustomerVector setPerformance(final double performance) {
		this.performance = performance;
		return this;
	}

	public int getClusterKey() {
		return clusterKey;
	}

	public CustomerVector setClusterKey(final int clusterKey) {
		this.clusterKey = clusterKey;
		return this;
	}

	public String getCustomerKey() {
		return customerKey;
	}

	public CustomerVector setCustomerKey(final String customerKey) {
		this.customerKey = customerKey;
		return this;
	}

    public double getEuclideanDist() {
		return euclideanDist;
	}

	public void setEuclideanDist(double euclideanDist) {
		this.euclideanDist = euclideanDist;
	}

	public String toString() {
    	return new StringBuilder("customerKey = ").append(customerKey).append(", clusterKey = ").append(String.valueOf(clusterKey)).append(", performance = ").append(String.valueOf(performance)).append(", dataPoints = ").append(delegate.toString()).toString();
    	
    }
}
