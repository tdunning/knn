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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.knn.cluster.BallKmeans;

public class StreamingKMeansReducer extends Reducer<IntWritable, ClusterWritable, IntWritable,
    ClusterWritable> {

  private BallKmeans clusterer;

  @Override
  public void setup(Context context) {

  }

  @Override
  public void reduce(IntWritable key, Iterable<ClusterWritable> centroids, Context context) {
  }

  @Override
  public void cleanup(Context context) {

  }
}