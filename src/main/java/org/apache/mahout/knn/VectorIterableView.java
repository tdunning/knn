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

import com.google.common.collect.Iterables;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;

import java.util.Iterator;

public class VectorIterableView implements VectorIterable {
    private VectorIterable data;
    private int start;
    private int rows;

    public VectorIterableView(VectorIterable data, int start, int rows) {
        this.data = data;
        this.start = start;
        this.rows = rows;
    }

    @Override
    public Iterator<MatrixSlice> iterateAll() {
        return Iterables.limit(Iterables.skip(data, start), rows).iterator();
    }

    @Override
    public int numSlices() {
        return rows;
    }

    @Override
    public int numRows() {
        return rows;
    }

    @Override
    public int numCols() {
        return this.iterateAll().next().vector().size();
    }

    /**
     * Return a new vector with cardinality equal to getNumRows() of this matrix which is the matrix product of the
     * recipient and the argument
     *
     * @param v a vector with cardinality equal to getNumCols() of the recipient
     * @return a new vector (typically a DenseVector)
     * @throws org.apache.mahout.math.CardinalityException
     *          if this.getNumRows() != v.size()
     */
    @Override
    public Vector times(Vector v) {
        throw new UnsupportedOperationException("Default operation");
    }

    /**
     * Convenience method for producing this.transpose().times(this.times(v)), which can be implemented with only one pass
     * over the matrix, without making the transpose() call (which can be expensive if the matrix is sparse)
     *
     * @param v a vector with cardinality equal to getNumCols() of the recipient
     * @return a new vector (typically a DenseVector) with cardinality equal to that of the argument.
     * @throws org.apache.mahout.math.CardinalityException
     *          if this.getNumCols() != v.size()
     */
    @Override
    public Vector timesSquared(Vector v) {
        throw new UnsupportedOperationException("Default operation");
    }

    @Override
    public Iterator<MatrixSlice> iterator() {
        return iterateAll();
    }
}
