package org.apache.mahout.knn.legacy;

import com.google.common.collect.Sets;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;

public class KMeansTest {
	
	public static List <DenseVector>list = new ArrayList<DenseVector>(100);
	public static List <DenseVector>centroidList = new ArrayList<DenseVector>(5);
	public static List <DenseVector>projectionList = new ArrayList<DenseVector>(5);
	public static TreeSet<Vector> s;
	
	static {
		final DenseVector projection = new DenseVector(10);
		final DoubleFunction random = Functions.random();
	    projection.assign(random);
	    projection.normalize();
	
	    s = Sets.newTreeSet(new Comparator<Vector>() {
	        @Override
	        public int compare(Vector v1, Vector v2) {
	            int r = Double.compare(v1.viewPart(3,10).dot(projection), v2.viewPart(3,10).dot(projection));
	            if (r == 0) {
	                return v1.hashCode() - v2.hashCode();
	            } else {
	                return r;
	            }
	        }
	    });
	
	}
	public static void createTestFile (String fileName) throws Exception {
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
	
	public static double euclideanDistance(Vector v1, Vector v2) {
		double edist = 0.0;
		int vlen = v1.size(); 
		for (int i = 0; i <vlen; i++){
			edist += Math.pow(v1.getQuick(i)-v2.getQuick(i),2);
		}
		return edist;
	}
	
    public static void main(String[] args) throws Exception { 
    	if (args.length == 1) {
    		createTestFile(args[0]);
    	}

    	FileReader fileReader=new FileReader(new File(args[0]));
    	BufferedReader bufferedReader=new BufferedReader(fileReader);
     	String [] values=bufferedReader.readLine().split(",");
    	double [] doubleValues=new double[values.length+2];
    	//Performance
    	doubleValues[1]=0;
    	//Centroid
    	doubleValues[2]=-1;
    	int counter = 0;
     	while ((values=bufferedReader.readLine().split(",")) != null) {
	     	for (int i=1; i < values.length; i++) {
	     		doubleValues[i+2]=Double.parseDouble(values[i]);
	     	}
	    	doubleValues[0]=Double.parseDouble(values[0]);
	 		list.add(new DenseVector(doubleValues));
	 		
	 		counter++;
	 		if(counter<5) {
		 		//weight of centroid;
		 		doubleValues[1]=1;
	 			s.add(new DenseVector(doubleValues));
	 		}
     	}
     	
    	fileReader.close();
    	int nobs = 100;
    	int k = 5;
    	int swapvector = -1;
    	double dotpvalue = 2000;
    	double f = (1/(k+Math.log(nobs)));
    	double mid1 = 0;
    	double mid2 = 0;
    	final DoubleFunction random = Functions.random();

    	
    	for (Vector vector : list) {
    		double edist=2000;
    		Vector centroidList [] = s.toArray(new Vector[s.size()]);
    		for (int i = 0; i < centroidList.length; i++) {
    			 mid1 = euclideanDistance(vector.viewPart(3,10),centroidList[i].viewPart(3,10));
    			 if (mid1 < edist) {
    				 edist = mid1;
    			 }
    		}
    	
    		if (Math.random()< (edist/f) ) {
                s.add(vector);
    			
   				Vector [] vectors = s.toArray(new Vector[s.size()]);
   				Vector compVector=new DenseVector(10);
   				compVector.assign(1000);
   				double minDist=1000;
   				double centroid1 = -1;
   				double centroid2 = 0;
   				double md = 0;
   				for (int i=0; i < vectors.length-1; i++) {
   					
   					md = euclideanDistance(vectors[i].viewPart(3,10),vectors[i+1].viewPart(3,10));
   					if (md < minDist) {
   						minDist=md;
   						centroid1 = vectors[i].getQuick(0);
   						centroid2 = vectors[i+1].getQuick(0);
   					}
   				}
    		}
    	}
    }

}
