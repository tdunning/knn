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
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.*;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.apache.mahout.math.random.WeightedThing;
import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class StreamingKMeansTestMR {
  private  MapDriver<IntWritable, CentroidWritable, IntWritable, CentroidWritable> mapDriver;

  public static final double[][] points = {
      {1, 1}, {2, 1}, {1, 2},
      {2, 2}, {3, 3}, {8, 8},
      {9, 8}, {8, 9}, {9, 9},
      {1.01, 1.0002}, {8.9999999, 8}
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
  public static void writePointsToFile(List<Centroid> points, String fileName, FileSystem fs,
                                       Configuration conf) throws IOException {
    Path path = new Path(fileName);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
        LongWritable.class, CentroidWritable.class);
    long recNum = 0;
    CentroidWritable writable = new CentroidWritable();
    for (Centroid point : points) {
      writable.setCentroid(point);
      writer.append(new LongWritable(recNum++), writable);
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
    int numClusters = 8;
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
    // writePointsToFile(vectors, "testdata/points/file1", fs, conf);

    // Run the KMeans algorithm.
    StreamingKMeansDriver.configureOptionsForWorkers(conf, 2, EuclideanDistanceMeasure.class
        .getName(), ProjectionSearch.class.getName(), 4, 3);
    StreamingKMeansDriver.run(conf, new Path("testdata/points/"), new Path("output/"));

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

  @Before
  public void setUp() {
    StreamingKMeansMapper mapper = new StreamingKMeansMapper();
    mapDriver = MapDriver.newMapDriver(mapper);
    Configuration conf = new Configuration();
    conf.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, EuclideanDistanceMeasure.class.getName());
    conf.setInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, 5);
    conf.setInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, 2);
    conf.set(StreamingKMeansDriver.SEARCHER_CLASS_OPTION, ProjectionSearch.class.getName());
    conf.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 2);
    mapDriver.setConfiguration(conf);
  }

  public static Iterable<Centroid> getCentroidFromRawArray(final double[][] points) {
    return new Iterable<Centroid>() {
      @Override
      public Iterator<Centroid> iterator() {
        return new Iterator<Centroid>() {
          private int i = 0;
          @Override
          public boolean hasNext() {
            return i < points.length;
          }

          @Override
          public Centroid next() {
            Centroid centroid = new Centroid(i, new DenseVector(points[i]), 1);
            ++i;
            return centroid;
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
      }
    };
  }

  @Test
  public void testMapper() throws IOException {
    double initialDistanceCutoff = DataUtils.estimateDistanceCutoff(getCentroidFromRawArray
        (points));
    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 2, 5), 2, initialDistanceCutoff);
    clusterer.cluster(getCentroidFromRawArray(points));

    for (Centroid centroid : getCentroidFromRawArray(points)) {
      mapDriver.addInput(new IntWritable(), new CentroidWritable(centroid));
    }
    for (Centroid centroid : clusterer.getCentroidsIterable()) {
      mapDriver.addOutput(new IntWritable(0), new CentroidWritable(centroid));
    }
    mapDriver.runTest();
  }

  @Test
  public void testHypercubeMapper() throws IOException {
    int numDimensions = 3;
    int numVertices = 1 << numDimensions;
    int numPoints = 1000;
    Pair<List<Centroid>, List<Centroid>> data =
        DataUtils.sampleMultiNormalHypercube(numDimensions, numPoints);
    for (Centroid point : data.getFirst()) {
      mapDriver.addInput(new IntWritable(), new CentroidWritable(point));
    }
    BruteSearch searcher = new BruteSearch(new EuclideanDistanceMeasure());
    int i = 0;
    for (Vector mean : data.getSecond()) {
      searcher.add(new WeightedVector(mean, 1, i++));
    }
    assertThat(i, equalTo(numVertices));
    int counts[] = new int[numVertices];
    List<org.apache.hadoop.mrunit.types.Pair<IntWritable,CentroidWritable>> results = mapDriver.run();
    for (org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable> result : results) {
      WeightedThing<Vector> closest = searcher.search(result.getSecond().getCentroid(), 1).get(0);
      assertThat(closest.getWeight(), is(Matchers.lessThan(0.05)));
      i = ((WeightedVector)closest.getValue()).getIndex();
      counts[i]++;
    }
    for (i = 0; i < numVertices; ++i) {
      assertThat(counts[i], equalTo(numPoints / numVertices));
    }
  }

  @Test
  public void testHypercubeSampleMR() throws IOException, ClassNotFoundException, InterruptedException {
    Configuration conf = new Configuration();
    int numDimensions = 3;
    int numDatapoints = 10000;
    int numClusters = 1 << numDimensions;
    String inPath = "cube-test/in-points";
    File testData = new File(inPath);
    if (!testData.exists()) {
      testData.mkdirs();
    }

    Pair<List<Centroid>, List<Centroid>> sample =
        DataUtils.sampleMultiNormalHypercube(numDimensions, numDatapoints);

    FileSystem fs = FileSystem.get(conf);
    // Write the generated vectors to a sequence file.
    writePointsToFile(sample.getFirst(), inPath, fs, conf);

    // Run the KMeans algorithm.
    StreamingKMeansDriver.configureOptionsForWorkers(conf, 2, EuclideanDistanceMeasure.class
        .getName(), ProjectionSearch.class.getName(), 4, 3);
    StreamingKMeansDriver.run(conf, new Path("testdata/points/"), new Path("output/"));

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
