package org.apache.mahout.knn.means;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.knn.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

public class KMeans {
	
	private static List <DenseVector>list = new ArrayList<DenseVector>(100);
	private static List <Centroid>centroidList = new ArrayList<Centroid>(6);
	private static int NUMBER_OF_CLUSTERS=5;
	private static String INPUT_FILE_NAME="C:\\kMeansTestFile.csv";
	private static String OUTPUT_FILE_NAME="C:\\kMeansOutFile.csv";
	private static final double BETA=1.5;
	private static int numOfRecordsRead=0;
	

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
	
	private final double euclideanDistance(Vector v1, Vector v2) {
		double edist = 0.0;
		int vlen = v1.size(); 
		for (int i = 0; i <vlen; i++){
			edist += Math.pow(v1.getQuick(i)-v2.getQuick(i),2);
		}
		return edist;
	}
	
	private final void readInputFile(String fileName) throws Exception{
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
	 		list.add(new DenseVector(doubleValues));
	 		
	 		numOfRecordsRead++;
     	}
     	
    	fileReader.close();

	}
	
	private final void initializeClusters() {
		for (int i=0; i < NUMBER_OF_CLUSTERS; i++) {
	 		centroidList.add(new Centroid(i, list.get(i)));
		}

	}
	
	private final void getClusters () {
		
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

	
	private final void setClusters() throws Exception{
	   	File dataFile = new File(OUTPUT_FILE_NAME);
    	FileWriter fileWriter=new FileWriter(dataFile);
    	StringBuilder stringBuilder = new StringBuilder();
		for (int i = 0; i <NUMBER_OF_CLUSTERS; i++) {
			centroidList.get(i).setWeight(0);
		}
    	for (Vector vector : list) {
    		double edist=2000;
    		double mid1 = 0;
    		int minIndex = -1;
    		for (int i = 0; i < centroidList.size(); i++) {
    			 mid1 = euclideanDistance(vector,centroidList.get(i).getVector());
    			 if (mid1 < edist) {
    				 edist = mid1;
    				 minIndex=i;
    			 }
    		}
    		centroidList.get(minIndex).addWeight();
    		stringBuilder.append(minIndex).append(",");
    		for (int j=0; j <10; j++) {
    			stringBuilder.append(String.valueOf(vector.get(j))).append(",");
    			
    		}
    		stringBuilder.setCharAt(stringBuilder.length()-1,'\n');
    		fileWriter.write(stringBuilder.toString());
    		stringBuilder.setLength(0);
    	}
    	fileWriter.close();	
	}

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
    		
    		
    	}
    	
    	kMeans.initializeClusters();
    	kMeans.getClusters();
    	kMeans.setClusters();
    	
    	for (int i=0; i < centroidList.size(); i++) {
    		System.out.println(centroidList.get(i));
    	}
    	
    }



}
