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

package org.apache.mahout.knn.means;

import org.apache.mahout.knn.Centroid;
import org.apache.mahout.knn.QueryPoint;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;
import java.util.TreeMap;

public class KMeansClusterSearch {
	
	private static List <QueryPoint>queryPoints = new ArrayList<QueryPoint>();
	private static List <Centroid>centroidList = new ArrayList<Centroid>();
	private static Hashtable <Integer, List<QueryPoint>> customerHash = new Hashtable<Integer, List<QueryPoint>>();
	private static int NUMBER_OF_NEIGHBORS=100;
	private static String QUERY_FILE_NAME="C:\\kMeansTestFile.csv";
	private static String REFERENCE_FILE_NAME="C:\\kMeansOutFile.csv";
	private static String CLUSTER_FILE_NAME="C:\\kMeansClusterFile.csv";
	private int lastClusterIndex=0;
	
    TreeMap<QueryPoint, java.util.Vector<Object[]>> matches = new TreeMap<QueryPoint, java.util.Vector<Object[]>>(new Comparator<QueryPoint>() {
        @Override
        public int compare(QueryPoint q1, QueryPoint q2) {
            return q1.getCustomerKey().compareTo(q2.getCustomerKey());
        }
    });
    

	private double euclideanDistance(Vector v1, Vector v2) {
		double edist = 0.0;
		int vlen = v1.size(); 
		for (int i = 0; i <vlen; i++){
			edist += Math.pow(v1.getQuick(i)-v2.getQuick(i),2);
		}
		return edist;
	}
	
	private void loadClusterFile(String fileName) throws Exception{
    	FileReader fileReader=new FileReader(new File(fileName));
    	BufferedReader bufferedReader=new BufferedReader(fileReader);
    	String line=null;
     	String [] values=bufferedReader.readLine().split(",");
    	double [] doubleValues=new double[values.length-2];
    	
     	while ((line=bufferedReader.readLine()) != null) {
     		values=line.split(",");
	     	for (int i=0; i < doubleValues.length; i++) {
	     		doubleValues[i]=Double.parseDouble(values[i+2]);
	     	}
			centroidList.add(new Centroid(Integer.valueOf(values[0]), new DenseVector(doubleValues), Integer.valueOf(values[1])));
	 		
     	}
     	
    	fileReader.close();

	}
	
	private void loadQueryFile(String fileName) throws Exception{
    	FileReader fileReader=new FileReader(new File(fileName));
    	BufferedReader bufferedReader=new BufferedReader(fileReader);
    	String line=null;
     	String [] values=bufferedReader.readLine().split(",");
    	double [] doubleValues=new double[values.length-1];
    	
     	while ((line=bufferedReader.readLine()) != null) {
     		values=line.split(",");
	     	for (int i=0; i < doubleValues.length; i++) {
	     		doubleValues[i]=Double.parseDouble(values[i+1]);
	     	}
			queryPoints.add(new QueryPoint(values[0], new DenseVector(doubleValues)));
	 		
     	}
     	
    	fileReader.close();

	}

	private void queryClusters() throws Exception{
		int centroidListSize=centroidList.size();
		
		List <double[]> customerDistances=null;
		double [] tuple;
		Centroid centroid;
    	for (QueryPoint queryPoint : queryPoints) {
    		tuple  = new double [3];
    		customerDistances=new ArrayList<double[]>(centroidListSize);
    		for (int i = 0; i < centroidListSize ; i++) {
    			centroid=centroidList.get(i);
    			tuple[0] = euclideanDistance(queryPoint.getDataPoints(),centroid.getVector());
    			tuple[1] = centroid.getKey();
    			tuple[2] = centroid.getWeight();
    		}
    		
    		Collections.sort(customerDistances,new Comparator<double[]>() {
                @Override
                public int compare(double[] d1, double[] d2) {
                	return Double.compare(d1[0], d2[0]);
                }
            }); 
    		
    		int weight=0, clusterIndex=0;
    		int customerDistancesSize=customerDistances.size();
    		for (; clusterIndex < customerDistancesSize; clusterIndex++) {
    			weight += customerDistances.get(clusterIndex)[2];
    			if (weight >= NUMBER_OF_NEIGHBORS) break;
    		}
    		if (weight < NUMBER_OF_NEIGHBORS) clusterIndex--;
    		
    		List<QueryPoint> refCustomerList;
    		java.util.Vector<Object[]> refFileDistances=new java.util.Vector<Object[]>();
    		
    		Object[] objectTuple;
    		
    		Vector queryPointVector= queryPoint.getDataPoints();
    		for (int i=0; i <= clusterIndex; i++) {
    			refCustomerList=customerHash.get(customerDistances.get(i)[1]);
    			objectTuple=new Object[2];
    			for (QueryPoint customer : refCustomerList) {
    				objectTuple[0]=euclideanDistance(queryPointVector,customer.getDataPoints());
    				objectTuple[1]=customer.getCustomerKey();
    				refFileDistances.add(objectTuple);
    			}
    		}
    		
    		//Sort  the list of distances we just got
        	Collections.sort(refFileDistances,new Comparator<Object[]>() {
                   @Override
                   public int compare(Object[] d1, Object[] d2) {
                   	return Double.compare((Double)d1[0], (Double)d2[0]);
                   }
            }); 

        	refFileDistances.setSize(NUMBER_OF_NEIGHBORS);
        	matches.put(queryPoint, refFileDistances);
        	     	
    	}
    		
    }

	
	private void loadReferenceFile(String fileName) throws Exception{
    	FileReader fileReader=new FileReader(new File(fileName));
    	BufferedReader bufferedReader=new BufferedReader(fileReader);
    	String line=null;
     	String [] values=bufferedReader.readLine().split(",");
    	double [] doubleValues=new double[values.length-3];
    	
    	String lastCluster="";
    	List <QueryPoint>customerList=null;
     	while ((line=bufferedReader.readLine()) != null) {
     		values=line.split(",");

     		for (int i=0; i < doubleValues.length; i++) {
	     		doubleValues[i]=Double.parseDouble(values[i+3]);
	     	}

	     	if (!values[0].equals(lastCluster)) {
	     		if (null != customerList) {
	     		 customerHash.put(Integer.valueOf(values[0]), customerList);
	     		}
	     		customerList = new ArrayList<QueryPoint>();
	     	}

	     	customerList.add(new QueryPoint(values[1], new DenseVector(doubleValues), Integer.valueOf(values[0]), Double.valueOf(values[2])));
	     	
	 		
     	}
     	
    	fileReader.close();

	}
	
	public static void main(String[] args) throws Exception {
    	
		KMeansClusterSearch kMeansClusterSearch=new KMeansClusterSearch();
    	
    	if (args.length == 0) {
    		kMeansClusterSearch.loadClusterFile(CLUSTER_FILE_NAME);
    		kMeansClusterSearch.loadReferenceFile(REFERENCE_FILE_NAME);
    		kMeansClusterSearch.loadQueryFile(QUERY_FILE_NAME);
    	} else if (args.length == 1) {
    		//NUMBER_OF_CLUSTERS=Integer.parseInt(args[0]);
    		//kMeans.createTestFile(INPUT_FILE_NAME);
    		//kMeans.readInputFile(INPUT_FILE_NAME);
    	} else if (args.length == 2) {
    		//NUMBER_OF_CLUSTERS=Integer.parseInt(args[0]);
    		//kMeans.readInputFile(args[1]);
    	} else if (args.length == 3) {
    		//NUMBER_OF_CLUSTERS=Integer.parseInt(args[0]);
    		//kMeans.readInputFile(args[1]);
    		//OUTPUT_FILE_NAME=args[2];
    		
    		
    	}
    	
    	kMeansClusterSearch.queryClusters();
//		kMeansClusterSearch.writeCustomerNeighbors();
		
    	
    }



}
