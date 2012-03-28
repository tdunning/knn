package org.apache.mahout.knn.means;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.mahout.knn.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

public class KMeans {
	
	private static List <DenseVector>list = new ArrayList<DenseVector>(100);
	private static List <Centroid>centroidList = new ArrayList<Centroid>(6);
	private static final int NUMBER_OF_CLUSTERS=5;
	private static final String DEFAULT_TEST_FILE_NAME="C:\\kMeansTestFile.csv";
	private static final double BETA=1.5;
	private static int numOfRecordsRead=0;
	

	private void createTestFile (String fileName) throws Exception {
	   	File dataFile = new File(fileName);
    	FileWriter fileWriter=new FileWriter(dataFile);
    	StringBuilder stringBuilder=new StringBuilder();
    	for(int i =0; i<100;i++) {
    		stringBuilder.append(i).append(",");
    		for (int j=0; j <10; j++) {
    			stringBuilder.append(Math.random()).append(",");
    		}
    		stringBuilder.setCharAt(stringBuilder.length()-1,'\n');
    		fileWriter.write(stringBuilder.toString());
    		stringBuilder.setLength(0);
    	}
    	fileWriter.close();
 		
	}
	
	private double euclideanDistance(Vector v1, Vector v2) {
		double edist = 0.0;
		int vlen = v1.size(); 
		for (int i = 0; i <vlen; i++){
			edist += Math.pow(v1.getQuick(i)-v2.getQuick(i),2);
		}
		return edist;
	}
	
	private void readInputFile(String fileName) throws Exception{
    	FileReader fileReader=new FileReader(new File(fileName));
    	BufferedReader bufferedReader=new BufferedReader(fileReader);
    	String line=null;
     	String [] values=bufferedReader.readLine().split(",");
    	double [] doubleValues=new double[values.length-1];
    	
     	while ((line=bufferedReader.readLine()) != null) {
     		values=line.split(",");
	     	for (int i=1; i < doubleValues.length; i++) {
	     		doubleValues[i]=Double.parseDouble(values[i+1]);
	     	}
	    	doubleValues[0]=Double.parseDouble(values[0]);
	 		list.add(new DenseVector(doubleValues));
	 		
	 		numOfRecordsRead++;
     	}
     	
    	fileReader.close();

	}
	
	private void initializeClusters() {
		for (int i=0; i < NUMBER_OF_CLUSTERS; i++) {
	 		centroidList.add(new Centroid(i, list.get(i)));
		}

	}
	
	private void getClusters () {
		
    	int minIndex = -1;
    	double f = 1/(NUMBER_OF_CLUSTERS+Math.log(numOfRecordsRead));
    	double mid1 = 0;
    	final DoubleFunction random = Functions.random();

    	Centroid [] centroidArray=centroidList.toArray(new Centroid [0]);
    	for (Vector vector : list) {
    		double edist=2000;
    		for (int i = 0; i < centroidArray.length; i++) {
    			 mid1 = euclideanDistance(vector,centroidArray[i].getVector());
    			 if (mid1 < edist) {
    				 edist = mid1;
    				 minIndex=i;
    			 }
    		}
    	
    		if (Math.random() > (edist/f) ) {
    			
    			centroidList.get(minIndex).update(new Centroid(1, vector));
    			
    		} else {
    			
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
	       			  mid1 = euclideanDistance(centroidList.get(i).getVector(),centroidList.get(i+1).getVector());    				
		    			if(Math.random() > (edist/f)) {
		    				centroidList.get(i).update(centroidList.get(i + 1));
		    				centroidList.remove(i + 1);
		    				for(int j = 0; j<sizeIndex; j++) {
		    					centroidList.get(j).setKey(j);
		    				}
		    				break;
		    			}
	    			}	
		    		f = BETA *f;
    			}
    		}	
    	}
    }

	
    public static void main(String[] args) throws Exception { 
    	
    	KMeans kMeans=new KMeans();
    	
    	if (args.length == 0) {
    		kMeans.createTestFile(DEFAULT_TEST_FILE_NAME);
    		kMeans.readInputFile(DEFAULT_TEST_FILE_NAME);
    	} else {
    		kMeans.readInputFile(args[0]);
    	}
    	
    	kMeans.initializeClusters();
    	kMeans.getClusters();
    	
    	for (int i=0; i < centroidList.size(); i++) {
    		System.out.println(centroidList.get(i));
    	}
    	
    }



}
