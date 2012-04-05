package com.aexp.ims.knn.mr;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Date;
import java.util.Iterator;

import static java.lang.Math.pow;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.aexp.ims.knn.key.KNNGroupComparator;
import com.aexp.ims.knn.key.KNNKey;
import com.aexp.ims.knn.key.KNNPartitioner;

@SuppressWarnings("deprecation")
public class KNNScorerWeighted extends Configured implements Tool {

	private static final int SCORING_FILE_LENGTH=50000;
	private static final int DEVELOPMENT_DATA_LENGTH=500000;
	private static final int CUTOFF_PERCENTILE=25;
	private static final int CUTOFF=DEVELOPMENT_DATA_LENGTH * CUTOFF_PERCENTILE /100;

	static enum CUSTOM_COUNTERS { NUMBER_OF_UNIQUE_KEYS}


	public static class KNNScorerMapper extends MapReduceBase
                    implements Mapper<LongWritable, Text, KNNKey, Text> {
    	
    	/**
    	 * This array will be available for the life span of every Mapper
    	 */
    	private double [][] dd = new double[SCORING_FILE_LENGTH][30];
    	private String [] custNum = new String[SCORING_FILE_LENGTH];
   	
    	
		/**
		 * Distribute the customer file to be scored to every node.  
		 *  Read it into the dd array for fast access.
		 *   
		 * Layout of Scoring File:
		 * 
		 * 0>  cust_xref_id
		 * 1>  wgt
		 * 2>  prem	
		 * 3>  std_APACTTEN,
		 * 4>  std_APAVGSESPD3,
		 * 5>  std_APAVGSESPD12,
		 * 6>  std_APTOTREMT12,
		 * 7>  std_AS_AYR_CCSG_012,
		 * 8>  std_AS_AYR_CCSG_014,
		 * 9>  std_AS_AYR_CCSG_020,
		 * 10> std_AS_AYR_CCSG_110,
		 * 11> std_AS_AYR_CCSG_113,
		 * 12> std_AS_AYR_CCSG_114,
		 * 13> std_AS_AYR_CCSG_115,
		 * 14> std_BEACON,
		 * 15> std_CMTDNET,
		 * 16> std_CMTDPC10,
		 * 17> std_CMTPNNET, 
		 * 18> std_CMTPPC10,
		 * 19> std_CUBBAL01,
		 * 20> std_MR_PGM_TENURE_NBR, 
		 * 21> std_MR_STD_VEST_PTS,
		 * 22> std_MSFILE,
		 * 23> std_N_HOMEVALUE, 
		 * 24> std_N_PREMIUM_NEW,
		 * 25> std_PINCOME,
		 * 26> std_SOWYRDB
		 *
		 */
    	public void configure(JobConf jobConf) {
    		
        	BufferedReader cacheReader=null;
    	    try {
                cacheReader = new BufferedReader(
                        new FileReader(new File("valfull.csv")));
	        	String line;
	        	String [] stringValues;
                int row=0; //variables begin at column 3
                while ((line = cacheReader.readLine()) != null) {
                	stringValues = line.split(",");
                	//The customer number.
                	custNum[row]=stringValues[1].trim();
                	for(int column=0; column < dd[0].length ; column++) {
                		try {
                    	  dd[row][column] = Double.parseDouble(stringValues[column+3]);
                		} catch (NumberFormatException nfe) {
                			//If we get a bad field - throw away the entire record.
                			row-=1;
                			break;
                		}
                	}
                	row++;
                }
 
            } catch (Exception ex) {
            	ex.printStackTrace();
            } finally {
    	    	if (null != cacheReader) 
    	    		
    	    		try {
    	    			cacheReader.close();
    	    		} catch (Exception ex) {
    	            	ex.printStackTrace();
    	            }	
            }
    	}
    	
		/**
		 * Compute all Euclidean distances for each record in the scored file.  
		 *   
		 * Layout of full development dataset:
		 * 
		 * 0>  cust_xref_id, 
		 * 1>  prem, 
		 * 2>  std_APACTTEN, 
		 * 3>  std_APAVGSESPD3, 
		 * 4>  std_APAVGSESPD12,
		 * 5>  std_APTOTREMT12, 
		 * 6>  std_AS_AYR_CCSG_012, 
		 * 7>  std_AS_AYR_CCSG_014, 
		 * 8>  std_AS_AYR_CCSG_020, 
		 * 9>  std_AS_AYR_CCSG_110, 
		 * 10> std_AS_AYR_CCSG_113,
		 * 11> std_AS_AYR_CCSG_114, 
		 * 12> std_AS_AYR_CCSG_115, 
		 * 13> std_BEACON,
		 * 14> std_CMTDNET, 
		 * 15> std_CMTDPC10, 
		 * 16> std_CMTPNNET, 
		 * 17> std_CMTPPC10, 
		 * 18> std_CUBBAL01, 
		 * 19> std_MR_PGM_TENURE_NBR, 
		 * 20> std_MR_STD_VEST_PTS, 
		 * 21> std_MSFILE, 
		 * 22> std_N_HOMEVALUE,
		 * 23> std_N_PREMIUM_NEW, 
		 * 24> std_PINCOME, 
		 * 25> std_SOWYRDB
		 *
		 * 
		 * Scoring file/Development Data/Array Indexes cross reference:
		 * 
   		 * Field		     	 		svIdx	ddIdx	score position
   		 *  
   		 * cust_xref_id			 		0		0		
   		 * wgt					 		-		1
   		 * prem					 		1		2
         * STD_APAVGSESPD12 	   		4		5		1
         * STD_CMTDPC10  		  		15		16		2
         * STD_APAVGSESPD3 		  		3		4		3
         * STD_APTOTREMT12 		  		5		6		4
         * STD_CMTDNET 			  		14		15		5
         * STD_AS_AYR_CCSG_114 	  		11		12		6
         * STD_AS_AYR_CCSG_012 	  		6		7		7
         * STD_CUBBAL01 		  		18		19		8
         * STD_AS_AYR_CCSG_014    		7		8		9
         * STD_AS_AYR_CCSG_113    		10		11		10
         * STD_AS_AYR_CCSG_115    		12		13		11
         * STD_MR_STD_VEST_PTS    		20		21		12
         * STD_PINCOME 			  		24		25		13
         * STD_AS_AYR_CCSG_110    		9		10		14
         * STD_N_HOMEVALUE 		  		22		23		15
         * STD_CMTPPC10 		  		17		18		16
         * STD_AS_AYR_CCSG_020    		8		9		17
         * STD_MR_PGM_TENURE_NBR  		19 		20		18
         * STD_APACTTEN 		  		2		3		19
         * STD_SOWYRDB 			  		25		26		20
         * STD_N_PREMIUM_NEW 	  		23		24		21
         * STD_CMTPNNET 		  		16		17		22
         * STD_MSFILE 			  		21		22		23
         * STD_BEACON 			  		13		14		24
		 * 
		 */
        public void map(LongWritable key, Text record, 
        			OutputCollector <KNNKey, Text> output,
        			Reporter reporter)
                        throws IOException { 
        	
            String [] stringValues = record.toString().split(",");
            double [] sv = new double[stringValues.length-3];
            
//        	for(int column=0; column < sv.length ; column++) {
            int lengthMinus1=sv.length-1;
           	for(int column=0; column < lengthMinus1 ; column++) {
        		try {
        			sv[column] = Double.parseDouble(stringValues[column+3]);                    		
        		} catch (NumberFormatException nfe) {
        			return;
        		}
        	}

            for(int custNumIdx=0; custNumIdx < dd.length; custNumIdx++) {
            	
            	double [] ddRow=dd[custNumIdx];
            
	            double d = pow(
	            		pow(ddRow[0]-sv[0],2) + 
	            		pow(ddRow[1]-sv[1],2) +
	            		pow(ddRow[2]-sv[2],2) +
	            		pow(ddRow[3]-sv[3],2) +
	            		pow(ddRow[4]-sv[4],2) +
	            		0,.5);
	            
	            double e = pow(
	            		pow(ddRow[0]-sv[0],2) + 
	            		pow(ddRow[1]-sv[1],2) +
	            		pow(ddRow[2]-sv[2],2) +
	            		pow(ddRow[3]-sv[3],2) +
	            		pow(ddRow[4]-sv[4],2) +
	            		pow(ddRow[5]-sv[5],2) + 
	            		pow(ddRow[6]-sv[6],2) +
	            		pow(ddRow[7]-sv[7],2) +
	            		pow(ddRow[8]-sv[8],2) +
	            		pow(ddRow[9]-sv[9],2) +
	            		0, .5);                                                                                             
	
	
	            double f =  pow(
	            		pow(ddRow[0]-sv[0],2) + 
	            		pow(ddRow[1]-sv[1],2) +
	            		pow(ddRow[2]-sv[2],2) +
	            		pow(ddRow[3]-sv[3],2) +
	            		pow(ddRow[4]-sv[4],2) +
	            		pow(ddRow[5]-sv[5],2) + 
	            		pow(ddRow[6]-sv[6],2) +
	            		pow(ddRow[7]-sv[7],2) +
	            		pow(ddRow[8]-sv[8],2) +
	            		pow(ddRow[9]-sv[9],2) +
	            		pow(ddRow[10]-sv[10],2) +
	            		pow(ddRow[11]-sv[11],2) +
	            		pow(ddRow[12]-sv[12],2) +
	            		pow(ddRow[13]-sv[13],2) +
	            		pow(ddRow[14]-sv[14],2) +
	            		pow(ddRow[15]-sv[15],2) + 
	            		pow(ddRow[16]-sv[16],2) +
	            		pow(ddRow[17]-sv[17],2) +
	            		pow(ddRow[18]-sv[18],2) +
	            		pow(ddRow[19]-sv[19],2) +
	            		pow(ddRow[20]-sv[20],2) +
	            		pow(ddRow[21]-sv[21],2) +
	            		pow(ddRow[22]-sv[22],2) +
	            		pow(ddRow[23]-sv[23],2) +
	            		pow(ddRow[24]-sv[24],2) +
	            		pow(ddRow[25]-sv[25],2) +
	            		pow(ddRow[26]-sv[26],2) +
	            		pow(ddRow[27]-sv[27],2) +
	            		pow(ddRow[28]-sv[28],2) +
	            		pow(ddRow[29]-sv[29],2) +
	            		0, .5);                                                                                                      
	           
	            
	    	    output.collect(new KNNKey(custNum[custNumIdx],d/5+e/10+f/30), new Text(new StringBuilder(stringValues[0]).append(",").
	            		append(f).toString()));

            }
            
        }
    }

    /*
     * Output one record per customer number.
     * 
     * prem = 0
     * d = 1
     * e = 2
     * f = 3
     */
    public static class KNNScorerReducer extends MapReduceBase 
    	implements Reducer<KNNKey, Text, KNNKey, Text> {
    	
        public void reduce(KNNKey key, Iterator<Text> values, 
        		OutputCollector <KNNKey, Text> output, Reporter reporter)
                           throws IOException {

        	double wo=0, so=0, wd2=0, sd2=0, riskd=0, riskd2=0, prem, f;   

        	String [] stringValues = null;
            for (int i=0; i < CUTOFF && values.hasNext(); i++) {
            	stringValues=values.next().toString().split(",");
            	try {
	            	prem = Double.parseDouble(stringValues[0]);
	            	f = Double.parseDouble(stringValues[1]);
            	} catch (NumberFormatException nfe) {
            		continue;
            	}
	    		wo += 1/(f+0.00001);                                                                                                                                 
	    		so += (1/(f+0.00001))* prem; 
	    		wd2 += 1/(f*f + 0.00001);                                                                                                              
	    		sd2 += (1/(f*f + 0.00001))* prem;
            }
            
            riskd = wo != 0 ? so/wo : 0;
            riskd2 = wd2 != 0 ? sd2/wd2 : 0;

            output.collect(key, new Text(new StringBuilder(",").append(riskd).append(",").append(riskd2).toString()));
        }
        
    }

    public int run(String[] args) throws Exception {
    	
        long startTime = new Date().getTime();
    	
        Configuration conf=getConf();
        JobConf jobConf = new JobConf(conf, KNNScorerWeighted.class);        
        jobConf.setJarByClass(KNNScorerWeighted.class);
        
        //The scored dataset
        FileInputFormat.setInputPaths(jobConf, new Path(args[0]));

        //The final output file
        FileOutputFormat.setOutputPath(jobConf, new Path(args[1]));
        
        jobConf.setMapperClass(KNNScorerMapper.class);
        jobConf.setReducerClass(KNNScorerReducer.class);
        jobConf.setNumReduceTasks(2000);
        jobConf.setNumMapTasks(2000);

        jobConf.setInputFormat(TextInputFormat.class);
        jobConf.setOutputFormat(TextOutputFormat.class);

        jobConf.setOutputKeyClass(KNNKey.class);
        jobConf.setOutputValueClass(Text.class);
        jobConf.setOutputValueGroupingComparator(KNNGroupComparator.class);
        jobConf.setPartitionerClass(KNNPartitioner.class);
        jobConf.set("key.value.separator.in.input.line", ",");
        
        JobClient.runJob(jobConf);

        System.out.println("\n\n\nTotal execution time " + 
                (new Date().getTime() - startTime) /60000 + " minutes.\n\n\n");
        
        return 0;
        
    }
    
    public static void main(String[] args) throws Exception { 
        int res = ToolRunner.run(new Configuration(), new KNNScorerWeighted(), args);
        System.exit(res);
    }
    
}
