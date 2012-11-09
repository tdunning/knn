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

package org.apache.mahout.knn.experimental;

import com.google.common.base.Preconditions;
import org.apache.commons.cli2.validation.InvalidArgumentException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.*;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.MatrixSlice;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.List;

public class StreamingKMeansMapper extends Mapper<IntWritable, CentroidWritable,
    IntWritable,CentroidWritable> {
  private static final Logger log = LoggerFactory.getLogger(Mapper.class);

  private StreamingKMeans clusterer;

  public static UpdatableSearcher searcherFromConfiguration(Configuration conf) {
    DistanceMeasure distanceMeasure = null;
    try {
      distanceMeasure = (DistanceMeasure)Class.forName(conf.get(DefaultOptionCreator.
          DISTANCE_MEASURE_OPTION)).newInstance();
    } catch (Exception e) {
      log.info("Failed to instantiate distanceMeasure");
      return null;
    }

    int searchSize = 0;
    searchSize = conf.getInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, 0);
    Preconditions.checkArgument(searchSize > 0, "Invalid search size.");

    int numProjections = 0;
    numProjections = conf.getInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, 0);
    Preconditions.checkArgument(numProjections > 0, "Invalid number of projections.");

    UpdatableSearcher searcher = null;
    String searcherClass = conf.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION);
    try {
      if (searcherClass.equals(BruteSearch.class.getName())) {
        searcher = (UpdatableSearcher)Class.forName(searcherClass).getConstructor(DistanceMeasure
            .class).newInstance(distanceMeasure);
      } else if (searcherClass.equals(FastProjectionSearch.class.getName()) ||
          searcherClass.equals(ProjectionSearch.class.getName())) {
        searcher = (UpdatableSearcher)Class.forName(searcherClass).getConstructor(DistanceMeasure
            .class, int.class, int.class).newInstance(distanceMeasure, numProjections, searchSize);
      } else if (searcherClass.equals(LocalitySensitiveHashSearch.class.getName())) {
        searcher = (UpdatableSearcher)Class.forName(searcherClass).getConstructor(DistanceMeasure
            .class, int.class).newInstance(distanceMeasure, searchSize);
      } else {
        log.error("Unknown searcher class instantiation requested "  + searcherClass);
        throw new InstantiationException();
      }
    } catch (Exception e) {
      e.printStackTrace();
      log.info("Failed to instantiate searcher " + searcherClass);
      return null;
    }
    return searcher;
  }

  @Override
  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    UpdatableSearcher searcher;
    searcher = searcherFromConfiguration(conf);
    Preconditions.checkNotNull(searcher);
    int numClusters = conf.getInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 0);
    Preconditions.checkArgument(numClusters > 0, "No number of clusters specified.");
    clusterer = new StreamingKMeans(searcher, numClusters);
  }

  @Override
  public void map(IntWritable key, CentroidWritable point, Context context) {
    clusterer.cluster(point.getCentroid());
  }

  @Override
  public void cleanup(Context context) throws IOException, InterruptedException {
    for (Centroid centroid : clusterer.getCentroidsIterable()) {
      context.write(new IntWritable(0), new CentroidWritable(centroid));
    }
  }
}
