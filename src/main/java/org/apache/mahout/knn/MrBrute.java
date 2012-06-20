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

package org.apache.mahout.knn;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class MrBrute extends Configured implements Tool {
    public static class Map extends Mapper<LongWritable, Text, Text, DoubleWritable> {
        private static double threshold1, threshold2;
        private static int nVar, nCusts;
        private Text cust = new Text();
        private List<double[]> trainingList = new ArrayList<double[]>();
        PriorityQueue<Double[]> queue = new PriorityQueue<Double[]>(100000, new DoubleComparator());

        public void setup(Context context) {
            Configuration conf = context.getConfiguration();
            String tt1 = conf.get("thr1");
            threshold1 = Double.parseDouble(tt1);
            nCusts = (int) threshold1;
            String tt2 = conf.get("thr2");
            threshold2 = Double.parseDouble(tt2);
            nVar = (int) threshold2;
            String line;
            String[] tokens;
            Path[] cacheFiles = new Path[0];
            try {
                cacheFiles = DistributedCache.getLocalCacheFiles(conf);
            } catch (IOException ioe) {
                System.err.println("Caught exception while getting cached files: " + StringUtils.stringifyException(ioe));
            }
            try {
                BufferedReader fis = new BufferedReader(new FileReader(cacheFiles[0].toString()));
                while ((line = fis.readLine()) != null) {
                    tokens = line.split(",");
                    double[] trainingObs = new double[nVar + 1];
                    trainingObs[0] = Double.parseDouble(tokens[0]);
                    for (int i = 3; i < (nVar + 3); i++) {
                        trainingObs[i - 2] = Double.parseDouble(tokens[i]);
                    }
                    trainingList.add(trainingObs);
                }
            } catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file '" + cacheFiles[0] + "' : " + StringUtils.stringifyException(ioe));
            }
        }

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] tokens = line.split(",");
            cust.set(tokens[1]);
            int n = nCusts;
            queue.clear();
            int cntr = 0;
            double[] testingObs = new double[nVar + 1];
            for (int i = 3; i < (nVar + 3); i++) {
                testingObs[i - 3] = Double.parseDouble(tokens[i]);
            }
            double sumTarget = 0;
            double dist;
            for (double[] trainingObs : trainingList) {
                cntr++;
                double dist1 = 0;
                for (int i1 = 0; i1 < nVar; i1++) {
                    dist1 += (trainingObs[i1 + 1] - testingObs[i1]) * (trainingObs[i1 + 1] - testingObs[i1]);
                }
                Double[] d = new Double[2];
                dist = dist1;
                d[0] = trainingObs[0];
                d[1] = dist;
                queue.add(d);
                if (cntr > n) {
                    d = queue.remove();
                }
            }
            for (int j = 0; j < n; j++) {
                Double[] d = queue.remove();
                sumTarget += d[0];
            }
            double prediction = (sumTarget / n);
            context.write(cust, new DoubleWritable(prediction));
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("thr1", args[3]);
        conf.set("thr2", args[4]);
        Job job = new Job(conf);
        job.setJarByClass(MrBrute.class);
        job.setJobName("KNN");
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        job.setMapperClass(Map.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        DistributedCache.addCacheFile(new Path(args[0]).toUri(), job.getConfiguration());
        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));


        boolean success = job.waitForCompletion(true);
        return success ? 0 : 1;

    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new MrBrute(), args);
        System.exit(res);
    }

    public static class DoubleComparator implements Comparator<Double[]> {
        @Override
        public int compare(Double[] x, Double[] y) {
            if (x[1] > y[1]) {
                return -1;
            }
            if (x[1] < y[1]) {
                return 1;
            }
            return 0;
        }
    }
}
