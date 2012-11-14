/**
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

package org.apache.mahout.knn.experimental;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.WeightedVector;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class BasicKMeans {
  public static final double[][] points = {
      {1, 1}, {2, 1}, {1, 2},
      {2, 2}, {3, 3}, {8, 8},
      {9, 8}, {8, 9}, {9, 9}
  };

  /**
   * Writes a list of Vectors to a sequence file as pairs of (LongWritable,
   * VectorWritable) where LongWritable is the key (the record index in this case) and
   * VectorWritable is the value.
   *
   * @param points the list of vectors to be written.
   * @param fileName the path of the sequence file to be created.
   * @param fs the HDFS FileSystem object to write this file on.
   * @param conf the Hadoop configuration file to write this file on.
   * @throws IOException
   */
  public static void writePointsToFile(List<Vector> points, String fileName, FileSystem fs,
                                       Configuration conf) throws IOException {
    Path path = new Path(fileName);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
        LongWritable.class, VectorWritable.class);
    long recNum = 0;
    VectorWritable vec = new VectorWritable();
    for (Vector point : points) {
      vec.set(point);
      writer.append(new LongWritable(recNum++), vec);
    }
    writer.close();
  }

  /**
   * Creates a List<Vector> from a 2D array of raw vectors.
   *
   * @param raw the array of vectors to be converted.
   * @return
   */
  public static List<Vector> getPoints(double[][] raw) {
    List<Vector> points = Lists.newArrayList();
    for (int i = 0; i < raw.length; ++i) {
      double[] fr = raw[i];
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(vec);
    }
    return points;
  }

  public static void main(String args[]) throws Exception {
    // The number of clusters to be formed.
    int numClusters = 2;
    List<Vector> vectors = getPoints(points);

    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    testData = new File("testdata/points");
    if (!testData.exists()) {
      testData.mkdir();
    }

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    // Wrote the generated vectors to a sequence file.
    writePointsToFile(vectors, "testdata/points/file1", fs, conf);

    Path path = new Path("testdata/clusters/part-00000");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class,
        Kluster.class);
    // Prepare the initial set of centroids.
    for (int i = 0; i < numClusters; ++i) {
      Vector vec = vectors.get(i);
      Cluster cluster = new Kluster(vec, i, new EuclideanDistanceMeasure());
      writer.append(new Text(cluster.asFormatString(null)), cluster);
    }
    writer.close();

    // Run the KMeans algorithm.
    KMeansDriver.run(conf,
        new Path("testdata/points/"), new Path("testdata/clusters"), new Path("output/"),
        new EuclideanDistanceMeasure(),
        0.0001,  // convergenceDelta
        10,  // maxIterations
        true,  // runClustering
        0.01,  // clusterClassificationThreshold
        false  // runSequential (if false, runs as MapReduce
    );

    // Print out final results.
    SequenceFile.Reader reader = new SequenceFile.Reader(fs,
        new Path("output/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000"), conf);
    IntWritable key = new IntWritable();
    WeightedVectorWritable value = new WeightedVectorWritable();
    while (reader.next(key, value)) {
      System.out.println(value.toString() + " belongs to cluster " + key.toString());
    }
    reader.close();
  }
}
