package org.apache.mahout.knn.means;

import java.io.BufferedReader;
import java.io.File;
import java.io.Closeable;
import java.io.IOException;
import java.io.FileWriter;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.PriorityQueue;
import java.util.Date;
import java.util.Set;
import java.util.TreeMap;

import static java.lang.Math.pow;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import org.apache.mahout.knn.Centroid;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.QueryPoint;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

public class MapRKMeansQueryTest extends Configured implements Tool  { 
	

	
	
	
	@SuppressWarnings("deprecation")
	public static class QueryScorer extends MapReduceBase
    implements Mapper<LongWritable, Text, Text, Text> {
		
		private static Hashtable <Integer, List<QueryPoint>> customerHash = new Hashtable<Integer, List<QueryPoint>>();
		private static List <Centroid>centroidList = new ArrayList<Centroid>();
		private static int NUMBER_OF_NEIGHBORS;
		//Using Configure for reading two files
		public void configure(JobConf jobConf) {
    		//String inputBaseFile = jobConf.get("INPUT_FILE_NAME");
    		//String clusterFile = jobConf.get("CLUSTER_FILE_NAME");
    		NUMBER_OF_NEIGHBORS = (int)Double.parseDouble(jobConf.get("maxNK"));
        	BufferedReader bufferReader=null;
    	    try {
                bufferReader = new BufferedReader(new FileReader(new File("kMeansOutFile.csv")));
	        	String line;
	        	String [] values;
	        	int counter = 0;
	        	
	        	String lastCluster="";
	        	List <QueryPoint>customerList=null;
	        	customerList = new ArrayList<QueryPoint>();
	         	while ((line=bufferReader.readLine()) != null) {
	         		values=line.split(",");
	         		double [] doubleValues=new double[values.length-3];
	         		if (counter == 0) {
	         			lastCluster = values[0];
	         			customerList = new ArrayList<QueryPoint>();
	         		}

	         		for (int i=0; i < doubleValues.length; i++) {
	    	     		doubleValues[i]=Double.parseDouble(values[i+3]);
	    	     	}

	    	     	if (!values[0].equals(lastCluster)) {
	    	     		if (null != customerList) {
	    	     		 customerHash.put(Integer.valueOf(lastCluster), customerList);
	    	     		}
	    	     		lastCluster = values[0];
	    	     		customerList = new ArrayList<QueryPoint>();	    	     		
	    	     	}
	    	     	
	    	     	customerList.add(new QueryPoint(values[1], new DenseVector(doubleValues), Integer.valueOf(values[0]), Double.parseDouble(values[2])));
	    	     	
	    	 		counter++;
	         	}
	         	customerHash.put(Integer.valueOf(lastCluster), customerList);
 
            } catch (Exception ex) {
            	ex.printStackTrace();
            } finally {
    	    	if (null != bufferReader) 
    	    		
    	    		try {
    	    			bufferReader.close();
    	    		} catch (Exception ex) {
    	            	ex.printStackTrace();
    	            }	
            }
    	    BufferedReader bufferReader1=null;
    	    try {
                bufferReader1 = new BufferedReader(new FileReader(new File("kMeansClusterFile.csv")));
	        	String line;
	        	String [] values;
                while ((line = bufferReader1.readLine()) != null) {
             		values=line.split(",");
             		double [] doubleValues=new double[values.length-2];
        	     	for (int i=0; i < doubleValues.length; i++) {
        	     		doubleValues[i]=Double.parseDouble(values[i+2]);
        	     	}
        	     	centroidList.add(new Centroid(Integer.valueOf(values[0]), new DenseVector(doubleValues), Double.parseDouble(values[1])));
                }
 
            } catch (Exception ex) {
            	ex.printStackTrace();
            } finally {
    	    	if (null != bufferReader1) 
    	    		
    	    		try {
    	    			bufferReader1.close();
    	    		} catch (Exception ex) {
    	            	ex.printStackTrace();
    	            }	
            }
    	} //End of Job Configuration
		
		public void map(LongWritable key, Text nRecord, OutputCollector <Text, Text> output,Reporter reporter) throws IOException {
    		String [] stringValues = nRecord.toString().split(",");
    		double [] valuesa = new double[stringValues.length-1];
    		
    		int lengthMinus1=valuesa.length-1;
    		for(int i=0; i < lengthMinus1 ; i++) {
        		try {
        			valuesa[i] = Double.parseDouble(stringValues[i+1]);                    		
        		} catch (NumberFormatException nfe) {
        			return;
        		}
        	}
    		QueryPoint queryPoint=new QueryPoint(stringValues[0], new DenseVector(valuesa));
    		int centroidListSize=centroidList.size();
    		Vector queryPointVector= new DenseVector(valuesa);
    		double [] tuple;
    		Centroid centroid;
    		
    		List <double[]> customerDistances=new ArrayList<double[]>(centroidListSize);
    		for (int i = 0; i < centroidListSize ; i++) {
    			centroid=centroidList.get(i);
    			tuple  = new double [3];
    			tuple[0] = euclideanDistance(queryPointVector,centroid.getVector());
    			tuple[1] = centroid.getIndex();
    			tuple[2] = centroid.getWeight();
    			customerDistances.add(tuple);
    		}
    		Collections.sort(customerDistances,new Comparator<double[]>() {
                @Override
                public int compare(double[] d1, double[] d2) {
                	double dd1 = d1[0];
                	double dd2 = d2[0];
                	return Double.compare(dd1, dd2);
                }
            }); 
    		
    		int weight=0, clusterIndex=0;
    		int customerDistancesSize=customerDistances.size();
    		for (; clusterIndex < customerDistancesSize; clusterIndex++) {
    			weight += customerDistances.get(clusterIndex)[2];
    			if (weight >= (2000)) break;
    		}
    		java.util.Vector<Object[]> refFileDistances=new java.util.Vector<Object[]>();
    		
    		Object[] objectTuple;
    		
    		
    		for (int j=0; j <= clusterIndex; j++) {
    			if(customerDistances.get(j)[2] > 0){
	    			int hashCustIndex = (int)customerDistances.get(j)[1];
	    			List<QueryPoint> refCustomerList = new ArrayList<QueryPoint>();
	    			refCustomerList= customerHash.get(hashCustIndex);
	    			
	    			for (QueryPoint customer : refCustomerList) {
	    				objectTuple=new Object[2];
	    				objectTuple[0]=euclideanDistance(queryPointVector,customer.getDataPoints());
	    				objectTuple[1]=customer.getCustomerKey();
	    				refFileDistances.add(objectTuple);
	    			}
    			}
    		}
    		
    		//Sort  the list of distances we just got
        	Collections.sort(refFileDistances,new Comparator<Object[]>() {
                   @Override
                   public int compare(Object[] d1, Object[] d2) {
                	   //double dd1 = Double.parseDouble(d1[0].toString());
                	   //double dd2 = Double.parseDouble(d2[0].toString());
                   		return Double.compare((Double)(d1[0]), (Double)(d2[0]));
                   }
            }); 
        	//limit the number of neighbors to a specific user specified size
        	refFileDistances.setSize(100);
        	
        	//writing nearest neighbors to a file
        	StringBuilder stringBuilder=new StringBuilder();
        		
        		for (Object[] objectTuple1 : refFileDistances){
        			stringBuilder.append(objectTuple1[1]).append(",").append(objectTuple1[0]).append(",");
        			output.collect(new Text(queryPoint.getCustomerKey()), new Text(stringBuilder.toString()));
        			stringBuilder.setLength(0);
        		}
        	refFileDistances.clear();
			//output.collect(new Text(queryPoint.getCustomerKey()), new Text(stringBuilder.toString()));
		}//End of Map
		
		private final double euclideanDistance(Vector v1, Vector v2) {
			double edist = 0.0;
			int vlen = v1.size(); 
			for (int i = 0; i <vlen; i++){
				edist += Math.pow(v1.getQuick(i)-v2.getQuick(i),2);
			}
			return edist;
		}
		
	} //End of Mapper
	
	
	@SuppressWarnings("deprecation")
    public int run(String[] args) throws Exception {
    	
        long startTime = new Date().getTime();
    	
        Configuration conf=getConf();
        JobConf jobConf = new JobConf(conf, MapRKMeansQueryTest.class);
        //jobConf.set("CLUSTER_FILE_NAME", args[1]);
        //jobConf.set("INPUT_FILE_NAME", args[0]);
        jobConf.set("maxNK", args[2]);
        jobConf.setJarByClass(MapRKMeansQueryTest.class);
        
        //The Query dataset
        FileInputFormat.setInputPaths(jobConf, new Path(args[0]));

        //The final output file
        FileOutputFormat.setOutputPath(jobConf, new Path(args[1]));
        
        jobConf.setMapperClass(QueryScorer.class);
        jobConf.setNumReduceTasks(1);
        jobConf.setNumMapTasks(100);

        jobConf.setInputFormat(TextInputFormat.class);
        jobConf.setOutputFormat(TextOutputFormat.class);

        jobConf.setOutputKeyClass(Text.class);
        jobConf.setOutputValueClass(Text.class);
        //jobConf.set("key.value.separator.in.input.line", ",");
        
        JobClient.runJob(jobConf);

        System.out.println("\n\n\nTotal execution time " + 
                (new Date().getTime() - startTime) /60000 + " minutes.\n\n\n");
        
        return 0;
        
    }
    
    public static void main(String[] args) throws Exception { 
        int res = ToolRunner.run(new Configuration(), new MapRKMeansQueryTest(), args);
        System.exit(res);
    }



}
