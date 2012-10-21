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

import org.apache.mahout.math.DelegatingVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Decorates a vector with a floating point weight and an index.
 */
public class HashedVector extends DelegatingVector implements Comparable<HashedVector> {
    private static final int INVALID_INDEX = -1;
    private long hash;
    private int index;

    public HashedVector(Vector v, long hash, int index) {
        super(v);
        this.hash = hash;
        this.index = index;
    }

    public HashedVector(Vector v, Matrix projection, int index, long mask) {
        super(v);
        this.index = index;
        this.hash = mask & computeHash64(v, projection);
    }

    public static int computeHash(Vector v, Matrix projection) {
        int hash = 0;
        for (Element element : projection.times(v)) {
            if (element.get() > 0) {
                hash += 1 << element.index();
            }
        }
        return hash;
    }

    public static long computeHash64(Vector v, Matrix projection) {
        long hash = 0;
        for (Element element : projection.times(v)) {
            if (element.get() > 0) {
                hash += 1L << element.index();
            }
        }
        return hash;
    }

    public static HashedVector hash(Vector v, Matrix projection) {
        return hash(v, projection, INVALID_INDEX, 0);
    }

    public static HashedVector hash(Vector v, Matrix projection, int index, long mask) {
        return new HashedVector(v, projection, index, mask);
    }

    public int xor(HashedVector v) {
        return Long.bitCount(v.getHash() ^ hash);
    }

    public long getHash() {
        return hash;
    }


    @Override
    public int compareTo(HashedVector other) {
        if (this == other) {
            return 0;
        }
        if (hash == other.getHash()) {
            double diff = this.minus(other).norm(1);
            if (diff < 1e-12) {
                return 0;
            } else {
                for (Element element : this) {
                    int r = Double.compare(element.get(), other.get(element.index()));
                    if (r != 0) {
                        return r;
                    }
                }
                return 0;
            }
        } else {
            if (hash > other.getHash()) {
                return 1;
            } else {
                return -1;
            }
        }
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    @Override
    public String toString() {
        return String.format("index=%d, hash=%08x, v=%s", index, hash, getVector());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof HashedVector)) {
            return o instanceof Vector && this.minus((Vector) o).norm(1) == 0;
        }           else {
            HashedVector v = (HashedVector) o;
            return v.hash == this.hash && this.minus(v).norm(1) == 0;
        }
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (int) (hash ^ (hash >>> 32));
        return result;
    }
}
