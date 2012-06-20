package org.apache.mahout.knn.means;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.knn.Centroid;
import org.apache.mahout.knn.CustomerVector;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.search.Searcher;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

public class KMeans { 
	
	private static List <CustomerVector>customerList = new ArrayList<CustomerVector>();
	private static int NUMBER_OF_CLUSTERS=5;
	private static String INPUT_FILE_NAME="C:\\kMeansTestFile.csv";
	private static String OUTPUT_FILE_NAME="C:\\kMeansOutFile.csv";
	private static String CLUSTER_FILE_NAME="C:\\kMeansClusterFile.csv";
	private static final double BETA=1.5;
	private static int numOfRecordsRead=0;
	private DistanceMeasure distanceMeasure;
    private static Matrix customerMatrix;
    private double distanceCutoff;
    private static Searcher clusters;


	private final void createTestFile (String fileName) throws Exception {
	   	File dataFile = new File(fileName);
    	FileWriter fileWriter=new FileWriter(dataFile);
    	StringBuilder stringBuilder=new StringBuilder();
        final Random gen = new Random();
    	for(int i =0; i<100;i++) {
    		stringBuilder.append(i).append(",");
    		for (int j=0; j <10; j++) {
    			stringBuilder.append(gen.nextGaussian()).append(",");
    		}
    		stringBuilder.setCharAt(stringBuilder.length()-1,'\n');
    		fileWriter.write(stringBuilder.toString());
    		stringBuilder.setLength(0);
    	}
    	fileWriter.close();
 		
	}
	
	private final void readInputFile(String fileName) throws Exception{
    	FileReader fileReader=new FileReader(new File(fileName));
    	BufferedReader bufferedReader=new BufferedReader(fileReader);
    	String line=null;
     	String [] values;
    	double [] doubleValues=null;
    	
     	while ((line=bufferedReader.readLine()) != null) {
     		values=line.split(",");
        	doubleValues=new double[values.length-1];
	     	for (int i=0; i < doubleValues.length; i++) {
	     		doubleValues[i]=Double.parseDouble(values[i+1]);
	     	}
	 		customerList.add(new CustomerVector(values[0],new DenseVector(doubleValues)));
	 		
	 		numOfRecordsRead++;
     	}
     	
       	customerMatrix=new DenseMatrix(numOfRecordsRead, doubleValues.length);
     	
     	for (int i=0; i < numOfRecordsRead; i++) {
     		customerMatrix.viewRow(i).assign(customerList.get(i));
     	}
     		
     	
    	fileReader.close();
	}
	
	
    public Searcher cluster(DistanceMeasure distanceMeasure, Iterable<MatrixSlice> data, int maxClusters) {
        // initialize scale
        distanceCutoff = estimateCutoff(data);
        this.distanceMeasure = distanceMeasure;

        // cluster the data
        return clusterInternal(data, maxClusters, 1);
        
        
    }
    

    private UpdatableSearcher clusterInternal(Iterable<MatrixSlice> data, int maxClusters, int depth) {
        int width = data.iterator().next().vector().size();
        UpdatableSearcher centroids = new ProjectionSearch(width, distanceMeasure, 4, 10);

        // now we scan the data and either add each point to the nearest group or create a new group
        // when we get too many groups, then we need to increase the threshold and rescan our current groups
        Random rand = RandomUtils.getRandom();
        int n = 0;
        for (MatrixSlice row : data) {
            if (centroids.size() == 0) {
                // add first centroid on first vector
                centroids.add(Centroid.create(centroids.size(), row.vector()), centroids.size());
            } else {
                // estimate distance d to closest centroid
                WeightedVector closest = centroids.search(row.vector(), 1).get(0);

                if (rand.nextDouble() < closest.getWeight() / distanceCutoff) {
                    // add new centroid, note that the vector is copied because we may mutate it later
                    centroids.add(Centroid.create(centroids.size(), row.vector()), centroids.size());
                } else {
                    // merge against existing
                    Centroid c = (Centroid) closest.getVector();
                    centroids.remove(c);
                    c.update(row.vector());
                    centroids.add(c, c.getIndex());
                }
            }

            if (depth < 2 && centroids.size() > maxClusters) {
                //maxClusters = (int) Math.max(maxClusters, 10 * Math.log(n));
                // TODO does shuffling help?
                List<MatrixSlice> shuffled = Lists.newArrayList(centroids);
                Collections.shuffle(shuffled);
                centroids = clusterInternal(shuffled, maxClusters, depth + 1);
                // for distributions with sharp scale effects, the distanceCutoff can grow to
                // excessive size leading sub-clustering to collapse the centroids set too much.
                // This test prevents that collapse from getting too severe.
                if (centroids.size() > 0.1 * maxClusters) {
                    distanceCutoff *= 1.5;
                }
            }
            n++;
        }
        return centroids;
    }
    

    public double estimateCutoff(Iterable<MatrixSlice> data) {
        Iterable<MatrixSlice> top = Iterables.limit(data, 100);

        // first we need to have a reasonable value for what a "small" distance is
        // so we find the shortest distance between any of the first hundred data points
        distanceCutoff = Double.POSITIVE_INFINITY;
        for (List<WeightedVector> distances : new Brute(top).search(top, 2)) {
            if (distances.size() > 1) {
                final double x = distances.get(1).getWeight();
                if (x != 0 && x < distanceCutoff) {
                    distanceCutoff = x;
                }
            }
        }
        return distanceCutoff;
    }

	private final void setClusters() throws Exception{
		Vector vector;
    	for (CustomerVector customerVector : customerList) {
    		customerVector.setClusterKey(clusters.search(customerVector, 1).get(0).getIndex());   
    	}
    	
		Collections.sort(customerList,new Comparator<CustomerVector>() {
            @Override
            public int compare(CustomerVector q1, CustomerVector q2) {
            	return Double.compare(q1.getClusterKey(), q2.getClusterKey());
            }
        }); 
    	
    	FileWriter fileWriter=new FileWriter(new File(OUTPUT_FILE_NAME));
    	StringBuilder stringBuilder = new StringBuilder();
    	for (CustomerVector customerVector : customerList) {
    		vector=customerVector.getVector();
			stringBuilder.append(customerVector.getClusterKey()).append(",").append(customerVector.getCustomerKey()).append(",").append(customerVector.getPerformance()).append(",");
			int vectorSize=vector.size();
			for (int j=0; j < vectorSize; j++) {
				stringBuilder.append(String.valueOf(vector.get(j))).append(",");
				
			}
			stringBuilder.setCharAt(stringBuilder.length()-1,'\n');
			fileWriter.write(stringBuilder.toString());
			stringBuilder.setLength(0);
    	}
    	fileWriter.close();	
	}


	private final void writeClusters() throws Exception{
	   	File dataFile = new File(CLUSTER_FILE_NAME);
    	FileWriter fileWriter=new FileWriter(dataFile);
    	StringBuilder stringBuilder = new StringBuilder();
        for (MatrixSlice cluster : clusters) {
            Centroid centroid = (Centroid) cluster.vector();
            int vectorSize = centroid.size();
            stringBuilder.append(centroid.getIndex()).append(",").append(centroid.getWeight()).append(",");
            for (int j = 0; j < vectorSize; j++) {
                stringBuilder.append(String.valueOf(centroid.get(j))).append(",");

            }
            stringBuilder.setCharAt(stringBuilder.length() - 1, '\n');
            fileWriter.write(stringBuilder.toString());
            stringBuilder.setLength(0);
        }
    	fileWriter.close();	
	}

	
	public static void main(String[] args) throws Exception {
    	
    	KMeans kMeans=new KMeans();
    	
    	if (args.length == 0) {
    		kMeans.createTestFile(INPUT_FILE_NAME);
    		kMeans.readInputFile(INPUT_FILE_NAME);
    	} else if (args.length == 1) {
    		NUMBER_OF_CLUSTERS=Integer.parseInt(args[0]);
    		kMeans.createTestFile(INPUT_FILE_NAME);
    		kMeans.readInputFile(INPUT_FILE_NAME);
    	} else if (args.length == 2) {
    		NUMBER_OF_CLUSTERS=Integer.parseInt(args[0]);
    		kMeans.readInputFile(args[1]);
    	} else if (args.length == 3) {
    		NUMBER_OF_CLUSTERS=Integer.parseInt(args[0]);
    		kMeans.readInputFile(args[1]);
    		OUTPUT_FILE_NAME=args[2];
    	} else if (args.length == 4) {
    		NUMBER_OF_CLUSTERS=Integer.parseInt(args[0]);
    		kMeans.readInputFile(args[1]);
    		OUTPUT_FILE_NAME=args[2];
    		CLUSTER_FILE_NAME=args[3];
    	}
    	
    	clusters =kMeans.cluster(new SquaredEuclideanDistanceMeasure(), customerMatrix, NUMBER_OF_CLUSTERS);
    	kMeans.setClusters();
    	kMeans.writeClusters();

        for (MatrixSlice cluster : clusters) {
            System.out.println(cluster.vector().toString());
        }
    	
    }



}
