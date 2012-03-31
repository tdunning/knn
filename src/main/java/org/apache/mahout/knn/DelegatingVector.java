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

import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.util.Iterator;

/**
 * A delegating vector provides an easy way to decorate vectors with weights or id's and such while
 * keeping all of the Vector functionality.
 */
public class DelegatingVector extends AbstractVector {
    protected Vector delegate;

    protected DelegatingVector(int size) {
        super(size);
    }

    public DelegatingVector(Vector v) {
        super(v.size());
        delegate = v;
    }

    @Override
    protected Matrix matrixLike(int i, int i1) {
        throw new UnsupportedOperationException("Can't make a matrix like this");
    }

    @Override
    public boolean isDense() {
        return delegate.isDense();
    }

    @Override
    public boolean isSequentialAccess() {
        return delegate.isSequentialAccess();
    }

    @Override
    public Iterator<Element> iterator() {
        return delegate.iterator();
    }

    @Override
    public Iterator<Element> iterateNonZero() {
        return delegate.iterateNonZero();
    }

    @Override
    public double getQuick(int i) {
        return delegate.getQuick(i);
    }

    @Override
    public Vector like() {
        return delegate.like();
    }

    @Override
    public void setQuick(int i, double v) {
        delegate.setQuick(i, v);
    }

    @Override
    public int getNumNondefaultElements() {
        return delegate.getNumNondefaultElements();
    }

    public Vector getVector() {
        return delegate;
    }
}
