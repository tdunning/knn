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

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;

import java.util.Random;

/**
 * Samples from a multi-variate normal distribution.
 */
public class MultiNormal implements Sampler<Vector> {
    private final Random gen = new Random();
    private final int dimension;
    private final Matrix a;
    private final Vector offset;

    public MultiNormal(Matrix a) {
        this(a, null);
    }

    public MultiNormal(Vector diagonal) {
        this(new DiagonalMatrix(diagonal), null);
    }

    public MultiNormal(Vector diagonal, Vector offset) {
        this(new DiagonalMatrix(diagonal), offset);
    }

    public MultiNormal(Matrix a, Vector offset) {
        this.a = a;
        this.offset = offset;
        this.dimension = a.columnSize();
    }

    public MultiNormal(int dimension) {
        this.dimension = dimension;
        this.a = null;
        this.offset = null;
    }

    @Override
    public Vector sample() {
        Vector v = new DenseVector(dimension).assign(
                new DoubleFunction() {
                    @Override
                    public double apply(double ignored) {
                        return gen.nextGaussian();
                    }
                }
        );
        if (offset != null) {
            return a.times(v).plus(offset);
        } else {
            if (a != null) {
                return a.times(v);
            } else {
                return v;
            }
        }
    }
}
