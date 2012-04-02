package org.apache.mahout.knn.LSH;

import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.IntArrayList;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by IntelliJ IDEA.
 * User: asanka
 * Date: 3/28/12
 * Time: 11:52 PM
 * To change this template use File | Settings | File Templates.
 */
public class LocalitySensitiveHash {
    private DistanceMeasure distance;
    private List<Vector> trainingVectors = Lists.newArrayList();
    private IntArrayList intKeys = new IntArrayList();
    private ArrayList<Integer> displacementList  = Lists.newArrayList();
    private int[] displacementCount = new int[33];
    private final int searchSize = 2000;
    int h1;
    int h2;
    // this matrix of 32 random vectors is used to compute the Locality Sensitive Hash
    // we compute the dot product with these vectors using a matrix multiplication and then use just
    // sign of each result as one bit in the hash
    private Matrix ranHash;
    public LocalitySensitiveHash(DistanceMeasure distance, int nVar) {
        this.distance = distance;
        // for initializing vectors
        ranHash = new DenseMatrix(32, nVar);
        for (int j=0; j < 32; j++){
            for(int k=0; k < nVar; k++){
                ranHash.set(j,k,((Math.random()*2)-1));
            }
        }
    }
    public List<IndexVector> search(Vector testingObs, int numberOfNeighbors) {
        PriorityQueue<IndexVector> pq = new PriorityQueue<IndexVector>(10, Ordering.natural().reverse());

        int query = computeHash(testingObs);
        for (int i=0; i<=32; i++) {
        	displacementCount[i]=0;
        }
        displacementList.clear();
        for (int i = 0; i < intKeys.size(); i++) {
            int approximateDistance = Integer.bitCount(query ^ intKeys.get(i));
            displacementList.add(approximateDistance);
            displacementCount[approximateDistance] += 1;
        };

        h1 = 0;
        h2 = 0;
        for (int i=0; i<=32; i++) {
        	h1 += displacementCount[i];
        	if (h1 >= searchSize) {
        		h2 = i;
        		break;
        	}
        }
        for (int i = 0; i < displacementList.size(); i++) {
            // int approximateDistance = Integer.bitCount(query ^ intKeys.get(i));
            if (displacementList.get(i) <= h2) {
                double dist = distance.distance(testingObs, trainingVectors.get(i));
                pq.add(new IndexVector(trainingVectors.get(i), i, dist));
                while (pq.size() > numberOfNeighbors) {
                    pq.poll();
                }
            }
        }
        List<IndexVector> r = Lists.newArrayList(pq);
        Collections.sort(r, Ordering.natural().reverse());
        return r;
    }
    
    public int countVectors(Vector testingObs) {
        int query = computeHash(testingObs);
        int k = 0;
        for (int i = 0; i < intKeys.size(); i++) {
            int approximateDistance = Integer.bitCount(query ^ intKeys.get(i));
            if (approximateDistance <= h2) {
                k++;
            }
        }
        return k;
    }

    public void add(Vector v) {
        trainingVectors.add(v);
        intKeys.add(computeHash(v));

    }

    private int computeHash(Vector v) {
        int r = 0;
        for (Vector.Element element : ranHash.times(v)) {
            if (element.get() > 0) {
                r +=1 << element.index();
            }
        }
        return r;
    }
    
    public static class IndexVector implements Comparable<IndexVector> {
        private double distance;
        private int index;
        private Vector v;

        public IndexVector(Vector v, int index, double weight) {
            this.v = v;
            this.index = index;
            this.distance = weight;
        }

        public int getIndex() {
            return index;
        }

        public Vector getV() {
            return v;
        }

        @Override
        public int compareTo(IndexVector o) {
            int r = Double.compare(distance, o.distance);
            if (r == 0) {
                return hashCode() - o.hashCode();
            } else {
                return r;
            }
        }
    }
}
