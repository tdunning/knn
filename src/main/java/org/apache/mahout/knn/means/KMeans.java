package org.apache.mahout.knn.means;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.knn.Centroid;
import org.apache.mahout.knn.CustomerVector;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;

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
	
	private void readInputFile(String fileName) throws Exception{
    	FileReader fileReader=new FileReader(new File(fileName));
    	BufferedReader bufferedReader=new BufferedReader(fileReader);
    	String line=null;
     	String [] values;
    	double [] doubleValues=null;
    	
    	CustomerVector customerVector;
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
	
	
    public UpdatableSearcher cluster(DistanceMeasure distanceMeasure, Iterable<MatrixSlice> data, int maxClusters) {
        // initialize scale
        distanceCutoff = estimateCutoff(data);
        this.distanceMeasure = distanceMeasure;

        // cluster the data
        UpdatableSearcher centroids = clusterInternal(data, maxClusters);

        // how make a clean set of empty centroids to get ready for final pass through the data
        int width = data.iterator().next().vector().size();
        UpdatableSearcher updatableSearcher = new ProjectionSearch(width, distanceMeasure, 4, 10);
        for (MatrixSlice centroid : centroids) {
            Centroid c = new Centroid(centroid.index(), new DenseVector(centroid.vector()));
            c.setWeight(0);
            updatableSearcher.add(c, c.getIndex());
        }

        // then make a final pass over the data
        for (MatrixSlice row : data) {
            WeightedVector closest = updatableSearcher.search(row.vector(), 1).get(0);

            // merge against existing
            Centroid c = (Centroid) closest.getVector();
            updatableSearcher.remove(c);
            c.update(row.vector());
            updatableSearcher.add(c, c.getIndex());
        }
        return updatableSearcher;
    }
    
    
    private UpdatableSearcher clusterInternal(Iterable<MatrixSlice> data, int maxClusters) {
        int width = data.iterator().next().vector().size();
        UpdatableSearcher centroids = new ProjectionSearch(width, distanceMeasure, 4, 10);

        // now we scan the data and either add each point to the nearest group or create a new group
        // when we get too many groups, then we need to increase the threshold and rescan our current groups
        Random rand = RandomUtils.getRandom();
        for (MatrixSlice row : data) {
            if (centroids.size() == 0) {
                // add first centroid on first vector
                centroids.add(new Centroid(centroids.size(), row.vector()), 0);
            } else {
                // estimate distance d to closest centroid
                WeightedVector closest = centroids.search(row.vector(), 1).get(0);

                if (rand.nextDouble() < closest.getWeight() / distanceCutoff) {
                    // add new centroid, note that the vector is copied because we may mutate it later
                    centroids.add(new Centroid(centroids.size(), new DenseVector(row.vector())), centroids.size());
                } else {
                    // merge against existing
                    Centroid c = (Centroid) closest.getVector();
                    centroids.remove(c);
                    c.update(row.vector());
                    centroids.add(c, c.getIndex());
                }
            }

            if (centroids.size() > maxClusters) {
                distanceCutoff *= 1.5;
                // TODO does shuffling help?
                List<MatrixSlice> shuffled = Lists.newArrayList(centroids);
                Collections.shuffle(shuffled);
                centroids = clusterInternal(shuffled, maxClusters);
            }
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

	
	/*
	private final void getClusters () {
		
    	int minIndex = -1;
    	double f = 1/(NUMBER_OF_CLUSTERS+Math.log(numOfRecordsRead));
    	double mid1 = 0;
    	final DoubleFunction random = Functions.random();

    	Vector vector;
    	Centroid [] centroidArray=centroidList.toArray(new Centroid [0]);
    	for (QueryPoint queryPoint : customerList) {
    		vector=queryPoint.getDataPoints();
    		
    		double edist=2000;
    		
    		//After this loop, minIndex points to the "closest" centroid.
    		for (int i = 0; i < centroidArray.length; i++) {
    			 mid1 = distanceMeasure.distance(vector,centroidArray[i].getVector());
    			 if (mid1 < edist) {
    				 edist = mid1;
    				 minIndex=i;
    			 }
    		}
    	
    		//This provides the non-determinism
    		if (Math.random() > (edist/f) ) {
    			//Add it to the closest centroid
    			centroidList.get(minIndex).update(new Centroid(1, vector));
    			
    		} else {
    			//Create a new centroid.
    			centroidList.add(new Centroid(NUMBER_OF_CLUSTERS + 1, vector));
                final DenseVector projection = new DenseVector(10);
                projection.assign(random);
                projection.normalize();
  			
                //Sort based on dot product
    			Collections.sort(centroidList,new Comparator<Centroid>() {
                    @Override
                    public int compare(Centroid c1, Centroid c2) {
                        int r = Double.compare(c1.getVector().dot(projection), c2.getVector().dot(projection));
                        if (r == 0) {
                            return c1.hashCode() - c2.hashCode();
                        } else {
                            return r;
                        }
                    }
                }); 
        		
        		int sizeIndex=0;
    			while(centroidList.size() > NUMBER_OF_CLUSTERS){
        		
    				sizeIndex = centroidList.size()-1;
	    			for (int i=0; i < sizeIndex; i++) {
	       			  mid1 = distanceMeasure.distance(centroidList.get(i).getVector(),centroidList.get(i+1).getVector());    				
		    			if(Math.random() > (edist/f)) {
		    				centroidList.get(i).update(centroidList.get(i + 1));
		    				centroidList.remove(i + 1);
		    				for(int j = 0; j<sizeIndex; j++) {
		    					centroidList.get(j).setIndex(j);
		    				}
		    				break;
		    			}
	    			}	
		    		f = BETA *f;
    			}
    		}	
    	}
    }

	
	private final void setClusters() throws Exception{
		for (int i = 0; i <NUMBER_OF_CLUSTERS; i++) {
			centroidList.get(i).setWeight(0);
		}
		Vector vector;
    	for (QueryPoint queryPoint : customerList) {
    		double edist=2000;
    		double mid1 = 0;
    		int minIndex = -1;
    		vector=queryPoint.getDataPoints();
    		for (int i = 0; i < centroidList.size(); i++) {
    			 mid1 = distanceMeasure.distance(vector,centroidList.get(i).getVector());
    			 if (mid1 < edist) {
    				 edist = mid1;
    				 minIndex=i;
    			 }
    		}
    		centroidList.get(minIndex).addWeight();
    		queryPoint.setClusterKey(minIndex);
    	}
    	
		Collections.sort(customerList,new Comparator<QueryPoint>() {
            @Override
            public int compare(QueryPoint q1, QueryPoint q2) {
            	return Double.compare(q1.getClusterKey(), q2.getClusterKey());
            }
        }); 
    	
    	FileWriter fileWriter=new FileWriter(new File(OUTPUT_FILE_NAME));
    	StringBuilder stringBuilder = new StringBuilder();
    	for (QueryPoint queryPoint : customerList) {
    		vector=queryPoint.getDataPoints();
			stringBuilder.append(queryPoint.getClusterKey()).append(",").append(queryPoint.getCustomerKey()).append(",").append(queryPoint.getPerformance()).append(",");
			for (int j=0; j <10; j++) {
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
    	for (Centroid centroid : centroidList) {
    		Vector vector = centroid.getVector();
    		int vectorSize=vector.size();
    		stringBuilder.append(centroid.getKey()).append(",").append(centroid.getWeight()).append(",");
    		for (int j=0; j < vectorSize; j++) {
    			stringBuilder.append(String.valueOf(vector.get(j))).append(",");
    			
    		}
    		stringBuilder.setCharAt(stringBuilder.length()-1,'\n');
    		fileWriter.write(stringBuilder.toString());
    		stringBuilder.setLength(0);
    	}
    	fileWriter.close();	
	}
	*/
	public final static void main(String[] args) throws Exception { 
    	
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
    	
    	UpdatableSearcher clusters =kMeans.cluster(new SquaredEuclideanDistanceMeasure(), customerMatrix, NUMBER_OF_CLUSTERS);
    	/*
    	kMeans.initializeClusters();
    	kMeans.getClusters();
    	kMeans.setClusters();
    	kMeans.writeClusters();
    	*/
    	Iterator<MatrixSlice> clusterIterator = clusters.iterator();
    	while (clusterIterator.hasNext()) {
    		System.out.println(((CustomerVector)clusterIterator.next().vector()).toString());
    		}
    	
    }



}
