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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import org.apache.mahout.common.RandomUtils;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Samples from a multinomial distribution using a fast tree algorithm.
 */
public class Multinomial<T> implements Sampler<T>, Iterable<T> {
    private SearchTree<T> tree;
    private final Random rand;
    private static final double EPSILON = 1e-10;

    public Multinomial(Multiset<T> counts, int width) {
        Preconditions.checkArgument(counts.size() > 0, "Need some data to build sampler");
        rand = RandomUtils.getRandom();
        List<WeightedThing<T>> things = Lists.newArrayList();
        double n = counts.size();
        for (T t : counts.elementSet()) {
            things.add(new WeightedThing<T>(t, counts.count(t) / n));
        }
        init(width, things);
    }

    public Multinomial(int width, Iterable<WeightedThing<T>> things) {
        rand = RandomUtils.getRandom();
        init(width, Lists.newArrayList(things));
    }

    private void init(int width, List<WeightedThing<T>> things) {
        Collections.sort(things);

        // now convert to cumulative weights to help with encoding as a tree
        double sum = 0;
        for (WeightedThing<T> thing : things) {
            final double w = thing.getWeight();
            sum += w;
            if (sum > 1) {
                // only can happen with round-off errors.  Since we add numbers up smallest
                // first, this should be a very minor probability.
                sum = 1;
            }
            thing.setWeight(sum);
        }
        // avoid round-off errors
        things.get(things.size() - 1).setWeight(1);

        // this allows us to build a tree that will help us sample fast
        tree = buildTree(0, 1, things, width);
    }

    /**
     * Recursively builds a search tree.
     *
     * @param low    The low bound for the search for this tree
     * @param high   The high bound for the search of this tree
     * @param things A list of things to be included in this branch of the tree
     * @param width  Branching factor for the tree
     * @return A search tree which may be an interior node or a sub-tree.
     */
    private SearchTree<T> buildTree(double low, double high, List<WeightedThing<T>> things, int width) {
        Preconditions.checkArgument(things.size() > 0, "Can't construct a tree with nothing");
        Preconditions.checkArgument(low <= things.get(0).getWeight(), "First element is outside outside of correct range");
        Preconditions.checkArgument(high <= things.get(things.size() - 1).getWeight(), "Last element is outside of correct range");

        if (things.size() == 1) {
            return new Leaf<T>(things.get(0).getValue());
        } else if (things.size() == 2) {
            final WeightedThing<T> t0 = things.get(0);
            final WeightedThing<T> t1 = things.get(1);
            return new Triplet<T>(ImmutableList.of(t0.getValue(), t1.getValue()), t0.getWeight(), high + 1);
        } else if (things.size() == 3) {
            final WeightedThing<T> t0 = things.get(0);
            final WeightedThing<T> t1 = things.get(1);
            final WeightedThing<T> t2 = things.get(2);
            return new Triplet<T>(ImmutableList.of(t0.getValue(), t1.getValue(), t2.getValue()), t0.getWeight(), t1.getWeight());
        } else if (things.size() <= width && high - low < EPSILON) {
            // these items are squeezed into such a small space that we really don't have to
            // worry about the details.  Thus we just give them all equal (and very small)
            // probabilities.
            Node<T> r = new Node<T>();
            for (WeightedThing<T> thing : things) {
                r.add(new Leaf<T>(thing.getValue()));
            }
            return r;
        } else {
            // each sub-tree here will take a uniform chunk of probability space.
            // if that chunk has only one element in it, that element will be a leaf
            int base = 0;
            Node<T> r = new Node<T>();
            r.low = low;
            r.high = high;
            final double step = (high - low) / width;
            for (int i = 0; i < width; i++) {
                double cutoff = Math.min(1, low + step);
                int top = base;
                while (top < things.size() && things.get(top).getWeight() < cutoff) {
                    top++;
                }
                r.add(buildTree(low, cutoff, things.subList(base, top + 1), width));
                low = cutoff;
                base = top;
            }
            return r;
        }
    }

    public T sample() {
        final double p = rand.nextDouble();
        return sample(p);
    }

    public T sample(double p) {
        return tree.find(p);
    }

    @Override
    public Iterator<T> iterator() {
        return tree.iterator();
    }

    private static interface SearchTree<T> extends Iterable<T> {
        T find(double p);
    }

    private static class Node<T> implements SearchTree<T> {
        double low, high;
        final List<SearchTree<T>> children;

        public Node() {
            children = Lists.newArrayList();
        }

        public void add(SearchTree<T> node) {
            children.add(node);
        }

        public T find(double p) {
            if (p < 0) {
                p = 0;
            }
            if (p > 1) {
                p = 1;
            }
            int slot = (int) ((p - low) / (high - low) * children.size());
            if (slot == children.size()) {
                slot = slot - 1;
            }
            return children.get(slot).find(p);
        }

        @Override
        public Iterator<T> iterator() {
            return Iterables.concat(children).iterator();
        }
    }

    private static class Triplet<T> implements SearchTree<T> {
        final double p1;
        final double p2;
        final List<T> values;

        private Triplet(List<T> values, double p1, double p2) {
            this.values = values;
            this.p2 = p2;
            this.p1 = p1;
        }

        public T find(double p) {
            if (p < p1) {
                return values.get(0);
            } else if (p >= p2) {
                return values.get(2);
            } else {
                return values.get(1);
            }
        }

        @Override
        public Iterator<T> iterator() {
            return values.iterator();
        }
    }

    private static class Leaf<T> implements SearchTree<T> {
        final T value;

        public Leaf(T value) {
            this.value = value;
        }

        public T find(double p) {
            return value;
        }

        @Override
        public Iterator<T> iterator() {
            return Iterators.singletonIterator(value);
        }
    }
}
