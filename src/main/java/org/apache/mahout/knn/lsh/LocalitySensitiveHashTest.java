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
package org.apache.mahout.knn.LSH;

import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import org.apache.commons.collections.ListUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.WeightedEuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Collections;
import java.util.List;

public class LocalitySensitiveHashTest {
    @Test
    public void testSearch() throws Exception {
        int nVar = 10;
        final DistanceMeasure distance = new EuclideanDistanceMeasure();
        //WeightedEuclideanDistanceMeasure weightFunction = new WeightedEuclideanDistanceMeasure();
        //Vector w = new DenseVector(nVar);
        //w.assign(1);
        //w.viewPart(0, 5).assign(2);
        //w.viewPart(5, 5).assign(1);
        //weightFunction.setWeights(w);
        double d1 = 0;
        double d2 = 0;
        double d3 = 0;
        double t1 = 0;
        double t2 = 0;
        double t3 = 0;
        double tsim = 0.0;
        double sim;
        int numberOfNeighbors = 100;
        int nearest = 100;
        int sz ;
        int tsz = 0;
        //LocalitySensitiveHash lsh = new LocalitySensitiveHash(weightFunction, nVar);
        LocalitySensitiveHash lsh = new LocalitySensitiveHash(distance, nVar);
        List<Vector> randomNeighbor = Lists.newArrayList();
        List<Vector> orgNeighbor = Lists.newArrayList();
        List<Vector> ref = Lists.newArrayList();
        //final DoubleFunction random = Functions.random();
        List<Vector> inputList = readInputFile("/Users/dixu/Documents/Amex/kNN/kMeansTestFile.csv");
        for (int i = 0; i < 40000; i++) {
            //Vector v = inputList.get(i);
            //v.assign(random);
            lsh.add(inputList.get(i));
            ref.add(inputList.get(i));
            orgNeighbor.add(inputList.get(i));
        }
        randomNeighbor.addAll(ref.subList(0, numberOfNeighbors));
        
        long runningTime = 0;
        for (int i= 40100; i < (40100+nearest); i++){
            final Vector v = inputList.get(i);
            //v.assign(random);
            long time1 = System.nanoTime();
            List<LocalitySensitiveHash.IndexVector> rx = lsh.search(v, numberOfNeighbors);
            
            List<Vector> lshNeighbor = Lists.newArrayList();
            for (LocalitySensitiveHash.IndexVector observation : rx) {
                lshNeighbor.add(ref.get(observation.getIndex()));
                }
            long time2 = System.nanoTime();
            runningTime = runningTime + time2 - time1;
            
            sz = lsh.countVectors(v);
            
            Ordering<Vector> queryOrder = new Ordering<Vector>() {
                @Override
                public int compare(Vector v1, Vector v2) {
                    return Double.compare(distance.distance(v, v1), distance.distance(v, v2));
                }
            };
            Collections.sort(orgNeighbor, queryOrder);
            List<Vector> trueNeighbor = orgNeighbor.subList(0,numberOfNeighbors);
            List<Vector> intersection1 = ListUtils.intersection(trueNeighbor, lshNeighbor);
            sim = intersection1.size() / (double) numberOfNeighbors;

            for (int j = 0; j < numberOfNeighbors; j++) {
                d1 += distance.distance(v, lshNeighbor.get(j));
                d2 += distance.distance(v, randomNeighbor.get(j));
                d3 += distance.distance(v, trueNeighbor.get(j));
            }
            d1 = d1 / numberOfNeighbors;
            d2 = d2 / numberOfNeighbors;
            d3 = d3 / numberOfNeighbors;
            t1+=d1;
            t2+=d2;
            t3+=d3;
            tsim+=sim;
            tsz+=sz;
        }
        t1=t1 / nearest;
        t2=t2 / nearest;
        t3=t3 / nearest;
        tsim=tsim / nearest;
        tsz=tsz / nearest;
        System.out.printf("ave_search=%d ave_sim=%.2f trueNeighbor_dist=%.2f proxyNeighbor_dist=%.2f " +
        		"randomNeighbor_dist=%.2f \n", tsz, tsim, t3, t1, t2);
        System.out.printf("running time = %.2f seconds \n", runningTime / 1e9);
    }
    private List<Vector> readInputFile(String fileName) throws Exception{
        List<Vector> inputList = Lists.newArrayList();
        FileReader fileReader=new FileReader(new File(fileName));
        BufferedReader bufferedReader=new BufferedReader(fileReader);
        String line;
        String [] values=bufferedReader.readLine().split(",");
        double [] doubleValues=new double[values.length-1];
        while ((line=bufferedReader.readLine()) != null) {
            values=line.split(",");
            for (int i=0; i < doubleValues.length; i++) {
                doubleValues[i]=Double.parseDouble(values[i+1]);
            }
            inputList.add(new DenseVector(doubleValues));
        }
        fileReader.close();
        return inputList;
    }
}
