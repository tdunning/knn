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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.LumpyData;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.Set;

public class ProjectionSearchTest extends FastProjectionSearchTest {
    private static Matrix data;
    private static final int QUERIES = 20;
    private static final int SEARCH_SIZE = 300;
    private static final int MAX_DEPTH = 100;

    @Override
    public UpdatableSearcher getSearch(int n) {
        return new ProjectionSearch(new EuclideanDistanceMeasure(), n, 4, 20);
    }
}
