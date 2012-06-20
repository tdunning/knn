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

/**
 * A QueryPoint is a customer.  
 */
public class QueryPoint {
	private double performance=-1;
    private double euclideanDist=-1;
    private int clusterKey = -1;
    private String customerKey;
    private Vector dataPoints;

    public QueryPoint(final String customerKey, final Vector dataPoints, 
    		final int clusterKey, final double performance) {
        this.customerKey = customerKey;
        this.dataPoints = dataPoints.clone();
        this.clusterKey=clusterKey;
        this.performance=performance;
    }

    public QueryPoint(final String customerKey, final Vector dataPoints) {
        this.customerKey = customerKey;
        this.dataPoints = dataPoints.clone();
    }

    public QueryPoint(final String customerKey) {
        this.customerKey = customerKey;
    }

    public QueryPoint() {
    }
    
    public double getPerformance() {
		return performance;
	}

	public QueryPoint setPerformance(final double performance) {
		this.performance = performance;
		return this;
	}

	public int getClusterKey() {
		return clusterKey;
	}

	public QueryPoint setClusterKey(final int clusterKey) {
		this.clusterKey = clusterKey;
		return this;
	}

	public String getCustomerKey() {
		return customerKey;
	}

	public QueryPoint setCustomerKey(final String customerKey) {
		this.customerKey = customerKey;
		return this;
	}

	public Vector getDataPoints() {
		return dataPoints;
	}

	public QueryPoint setDataPoints(final Vector dataPoints) {
		this.dataPoints = dataPoints;
		return this;
	}

    public double getEuclideanDist() {
		return euclideanDist;
	}

	public void setEuclideanDist(double euclideanDist) {
		this.euclideanDist = euclideanDist;
	}

	public String toString() {
        return String.format("customerKey = %s, clusterKey = %d, performance = %f, dataPoints = %s",
                customerKey, clusterKey, performance, dataPoints);
    }
}
