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

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.google.common.base.Preconditions;
import org.apache.commons.math.ArgumentOutsideDomainException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.*;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.LocalitySensitiveHashSearch;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertThat;

/**
 * Classifies the vectors into different clusters found by the clustering
 * algorithm.
 */
public final class StreamingKMeansDriver extends AbstractJob {
  // TODO(dfilimon): These constants should move to DefaultOptionCreator and so should the code
  // that handles their creation.
  public static final String SEARCHER_CLASS_OPTION = "searcherClass";
  public static final String NUM_PROJECTIONS_OPTION = "numProjections";
  public static final String SEARCH_SIZE_OPTION = "searchSize";

  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansDriver.class);

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator
        .numClustersOption()
        .withDescription(
            "The k in k-Means. Approximately this many clusters will be generated.").create());
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption("searcherClass", "sc", "The type of searcher to be used when performing nearest " +
        "neighbor searches. Defaults to BruteSearch.", "org.apache.mahout.knn.search" +
        ".BruteSearch");
    addOption("numProjections", "np", "The number of projections considered in estimating the " +
        "distances between vectors. Only used when the distance measure requested is either " +
        "ProjectionSearch or FastProjectionSearch. If no value is given, defaults to 20.", "20");
    addOption("searchSize", "s", "In more efficient searches (non BruteSearch), " +
        "not all distances are calculated for determining the nearest neighbors. The number of " +
        "elements whose distances from the query vector is actually computer is proportional to " +
        "searchSize. If no value is given, defaults to 10.", "10");

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    configureOptionsForWorkers();

    run(getConf(), input, output);
    return 0;
  }

  /**
   * Set up the Configuration object used by Mappers and Reducers to configure
   * themselves with the requested options. The options control:
   * <ul>
   *   <li>how many clusters to generate</li>
   *   <li>which distance measure to use</li>
   *   <li>which searcher class to use, and what parameters to instantiate it with</li>
   * </ul>
   */
  private void configureOptionsForWorkers() {
    Configuration conf = getConf();
    log.info("Starting to configure options for workers");

    // The number of clusters to generate.
    String numClusters = getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION);
    if (numClusters == null) {
      throw new IllegalArgumentException("No number of clusters specified.");
    }
    conf.set(DefaultOptionCreator.NUM_CLUSTERS_OPTION, numClusters);

    // The distance measure class to use.
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = EuclideanDistanceMeasure.class.getName();
    }
    try {
      DistanceMeasure distanceMeasure = (DistanceMeasure)Class.forName(measureClass).newInstance();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (Exception e) {
    }
    conf.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, measureClass);

    // The searcher class to use. This should never be null because of the default value.
    String searcherClass = getOption(SEARCHER_CLASS_OPTION);
    Preconditions.checkNotNull(searcherClass);
    conf.set(SEARCHER_CLASS_OPTION, searcherClass);

    // If the searcher we'll be using is the basic BruteSearch, no other data is needed.
    if (searcherClass.equals(BruteSearch.class.getName())) {
      return;
    }

    // The search size to use. This is quite fuzzy and might end up not being configurable at all
    // . For now, it's available for experimentation. This will never be null because of the
    // default value.
    String searchSize = getOption(SEARCH_SIZE_OPTION);
    Preconditions.checkNotNull(searchSize);
    conf.set(SEARCH_SIZE_OPTION, searchSize);

    // If the searcher we'll be using is a locality sensitive hash searcher,
    // no other data is needed.
    if (searcherClass.equals(LocalitySensitiveHashSearch.class.getName())) {
      return;
    }

    // The number of projections to use. This will never be null because of the default value.
    String numProjections = getOption(NUM_PROJECTIONS_OPTION);
    Preconditions.checkNotNull(numProjections);
    conf.set(NUM_PROJECTIONS_OPTION, numProjections);
  }

  public static void configureOptionsForWorkers(Configuration conf, int numClusters,
                                                String measureClass, String searcherClass,
                                                int searchSize, int numProjections) {
    // conf.setC
    conf.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, numClusters);
    conf.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, measureClass);
    conf.set(SEARCHER_CLASS_OPTION, searcherClass);
    conf.setInt(SEARCH_SIZE_OPTION, searchSize);
    conf.setInt(NUM_PROJECTIONS_OPTION, numProjections);
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the results of the final iteration to
   * cluster the input vectors.
   *
   * @param input
   *          the directory pathname for input points
   * @param output
   *          the directory pathname for output points
   */
  public static void run(Configuration conf, Path input, Path output)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("Starting StreamingKMeans clustering");

    // Prepare Job for submission.
    Job job = new Job(conf, "StreamingKMeans");

    // Input and output file format.
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // Mapper output Key and Value classes.
    // We don't really need to output anything as a key, since there will only be 1 reducer.
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(CentroidWritable.class);

    // Reducer output Key and Value classes.
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(CentroidWritable.class);

    // Mapper and Reducer classes.
    job.setMapperClass(StreamingKMeansMapper.class);
    job.setReducerClass(StreamingKMeansReducer.class);

    // There is only one reducer so that the intermediate centroids get collected on one
    // machine and are clustered in memory to get the right number of clusters.
    job.setNumReduceTasks(1);

    // Set input and output paths for the job.
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    // Set the JAR (so that the required libraries are available) and run.
    job.setJarByClass(StreamingKMeansDriver.class);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("StreamingKMeans interrupted");
    }

    log.info("StreamignKMeans job complete");
  }

  /**
   * Constructor to be used by the ToolRunner.
   */
  private StreamingKMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new StreamingKMeansDriver(), args);
  }
}
