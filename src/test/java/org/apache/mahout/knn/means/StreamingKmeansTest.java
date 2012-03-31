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

package org.apache.mahout.knn.means;

import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.generate.MultiNormal;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;
import java.util.Random;


public class StreamingKmeansTest {
    @Test
    public void testEstimateBeta() {
        Matrix m = new DenseMatrix(8, 3);
        for (int i = 0; i < 8; i++) {
            m.viewRow(i).assign(new double[]{0.125 * (i & 4), i & 2, i & 1});
        }
        Assert.assertEquals(0.5, new StreamingKmeans().estimateBeta(m), 1e-9);
    }

    @Test
    public void testClustering1() {
        Matrix data = new DenseMatrix(800, 3);
        int k = 0;
        Matrix mean = new DenseMatrix(8, 3);
        List<MultiNormal> rowSamplers = Lists.newArrayList();
        for (int i = 0; i < 8; i++) {
            mean.viewRow(i).assign(new double[]{0.25 * (i & 4), 0.5 * (i & 2), i & 1});
            MultiNormal gen = new MultiNormal(0.1, mean.viewRow(i));
            rowSamplers.add(gen);
        }


        Random rowSelector = RandomUtils.getRandom();
        for (MatrixSlice slice : data) {
            slice.vector().assign(rowSamplers.get(rowSelector.nextInt(8)).sample());
        }


        ProjectionSearch r = new StreamingKmeans().cluster(new EuclideanDistanceMeasure(), data, 30);

    }
}
