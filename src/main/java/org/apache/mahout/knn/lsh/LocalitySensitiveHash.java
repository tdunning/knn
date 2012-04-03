package org.apache.mahout.knn.lsh;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.search.Searcher;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.TreeSet;


public class LocalitySensitiveHash extends Searcher implements Iterable<MatrixSlice> {
    private DistanceMeasure distance;
    private List<WeightedVector> trainingVectors;

    // private ArrayList<Integer> displacementList  = Lists.newArrayList();
    private int[] displacementCount = new int[33];
    private int searchSize;
    int h1;
    int h2;
    // this matrix of 32 random vectors is used to compute the Locality Sensitive Hash
    // we compute the dot product with these vectors using a matrix multiplication and then use just
    // sign of each result as one bit in the hash
    private Matrix ranHash;
    public LocalitySensitiveHash(DistanceMeasure distance, int nVar, int searchSize) {
        this.distance = distance;
        this.searchSize = searchSize;
        trainingVectors = Lists.newArrayList();
        // for initializing vectors
        ranHash = new DenseMatrix(32, nVar);
        for (int j=0; j < 32; j++){
            for(int k=0; k < nVar; k++){
                ranHash.set(j,k,((Math.random()*2)-1));
            }
        }
    }
    public List<WeightedVector> search(Vector testingObs, int numberOfNeighbors) {

        int query = computeHash(testingObs);
        for (int i=0; i<=32; i++) {
        	displacementCount[i]=0;
        }
        // displacementList.clear();
        for (WeightedVector v : trainingVectors) {
        	int training = (int)v.getWeight();
            int approximateDistance = Integer.bitCount(query ^ training);
            v.setIndex(approximateDistance);
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
        
        List<WeightedVector> top = Lists.newArrayList();

        for (WeightedVector v : trainingVectors) {
            if (v.getIndex() <= h2) {
                double dist = distance.distance(testingObs, v.getVector());
                top.add(new WeightedVector(v.getVector().clone(), dist, -1));
            }
        }

        // Collections.sort(top, byQueryDistance(testingObs));
        Collections.sort(top);
        return top.subList(0, numberOfNeighbors);
    }
    
    private Ordering<Vector> byQueryDistance(final Vector query) {
        return new Ordering<Vector>() {
            @Override
            public int compare(Vector v1, Vector v2) {
                int r = Double.compare(distance.distance(query, v1), distance.distance(query, v2));
                return r != 0 ? r : v1.hashCode() - v2.hashCode();
            }
        };
    }

    
    public int countVectors() {
        int k = 0;
        for (WeightedVector v : trainingVectors) {
            if (v.getIndex() <= h2) {
                k++;
            }
        }
        return k;
    }
    
    public void add(Vector v, int index) {
    	double weight = computeHash(v);
        trainingVectors.add(new WeightedVector(v,weight,index));
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
    
    public int size() {
        return trainingVectors.size();
    }
    
    @Override
    public int getSearchSize() {
        return searchSize;
    }

    @Override
    public void setSearchSize(int size) {
        searchSize = size;
    }

    @Override
    public Iterator<MatrixSlice> iterator() {
        return new AbstractIterator<MatrixSlice>() {
            int index = 0;
            Iterator<WeightedVector> data = trainingVectors.iterator();

            @Override
            protected MatrixSlice computeNext() {
                if (!data.hasNext()) {
                    return endOfData();
                } else {
                    return new MatrixSlice(data.next(), index++);
                }
            }
        };
    }
}
