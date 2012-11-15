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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.sun.istack.internal.Nullable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.knn.cluster.BallKMeans;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.WeightedVector;

import java.io.IOException;
import java.util.Iterator;

public class StreamingKMeansReducer extends Reducer<IntWritable, CentroidWritable, IntWritable,
    CentroidWritable> {

  private int numClusters;
  private int maxNumIterations;

  @Override
  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    numClusters = conf.getInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 0);
    if (numClusters < 1) {
      throw new RuntimeException("Number of clusters must be positive: " + numClusters);
    }
    maxNumIterations = conf.getInt(StreamingKMeansDriver.MAX_NUM_ITERATIONS, 0);
    if (maxNumIterations < 1) {
      throw new RuntimeException("Maximum number of iterations must be positive: " +
          maxNumIterations);
    }
  }

  @Override
  public void reduce(IntWritable key, Iterable<CentroidWritable> centroids,
                     Context context) throws IOException, InterruptedException {
    BallKMeans clusterer = new BallKMeans(numClusters, Iterables.transform(centroids,
        new Function<CentroidWritable, WeightedVector>() {
      @Override
      public WeightedVector apply(@Nullable CentroidWritable input) {
        return input.getCentroid();
      }
    }), maxNumIterations);
    Iterator<Centroid> ci = clusterer.iterator();
    while (ci.hasNext()) {
      Centroid centroid = ci.next();
      context.write(new IntWritable(centroid.getIndex()), new CentroidWritable(centroid));
    }
  }
}
