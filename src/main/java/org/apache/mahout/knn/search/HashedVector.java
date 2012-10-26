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

import com.google.common.base.Preconditions;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;

/**
 * Decorates a weighted vector with a locality sensitive hash.
 */
public class HashedVector extends WeightedVector {
  protected static int INVALID_INDEX = -1;
  private long hash;

  public HashedVector(Vector v, long hash, int index) {
    super(v, 1, index);
    this.hash = hash;
  }

  public HashedVector(Vector v, Matrix projection, int index, long mask) {
    super(v, 1, index);
    this.hash = mask & computeHash64(v, projection);
  }

  public HashedVector(WeightedVector v, Matrix projection, long mask) {
    super(v.getVector(), v.getWeight(), v.getIndex());
    this.hash = mask;
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

  public static HashedVector hash(WeightedVector v, Matrix projection) {
    return hash(v, projection, 0);
  }

  public static HashedVector hash(WeightedVector v, Matrix projection, long mask) {
    return new HashedVector(v, projection, mask);
  }

  public int xor(HashedVector v) {
    return Long.bitCount(v.getHash() ^ hash);
  }

  public long getHash() {
    return hash;
  }

  /*
  Implements Comparable<HashedVector> by getting a WeightedVector and checking its actual type at runtime.
   */
  @Override
  public int compareTo(WeightedVector other) {
    if (other instanceof WeightedVector)
      return super.compareTo(other);
    HashedVector hashedOther = (HashedVector)other;
    if (this == hashedOther) {
      return 0;
    }
    if (hash == hashedOther.getHash()) {
      double diff = this.minus(hashedOther).norm(1);
      if (diff < 1e-12) {
        return 0;
      } else {
        for (Element element : this) {
          int r = Double.compare(element.get(), hashedOther.get(element.index()));
          if (r != 0) {
            return r;
          }
        }
        return 0;
      }
    } else {
      if (hash > hashedOther.getHash()) {
        return 1;
      } else {
        return -1;
      }
    }
  }

  @Override
  public String toString() {
    return String.format("index=%d, hash=%08x, v=%s", getIndex(), hash, getVector());
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
