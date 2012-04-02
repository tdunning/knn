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

import com.google.common.collect.Lists;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Created by IntelliJ IDEA.
 * User: tdunning
 * Date: 4/2/12
 * Time: 10:11 AM
 * To change this template use File | Settings | File Templates.
 */
public class ThreadedMapReduce implements Mapper<Writable, Writable, Writable, Writable> {
    private Writable endMarker = new IntWritable();
    
    private ExecutorService pool;
    private final Queue<Writable> inputQueue = new ConcurrentLinkedQueue<Writable>();
    private final BlockingQueue<Writable> outputQueue = new LinkedBlockingQueue<Writable>();
    private OutputCollector<Writable, Writable> outputCollector;
    private List<Future<Writable>> futures;

    @Override
    public void map(Writable key, Writable value, OutputCollector<Writable, Writable> outputCollector, Reporter reporter) throws IOException {
        this.outputCollector = outputCollector;
        inputQueue.add(value);
        List<Writable> results = Lists.newArrayList();
        outputQueue.drainTo(results);
        for (Writable result : results) {
            outputCollector.collect(key, result);
        }
    }

    @Override
    public void close() throws IOException {
        for (int i = 0; i < 20; i++) {
            inputQueue.add(endMarker);
        }
        pool.shutdown();

        List<Writable> results = Lists.newArrayList();
        outputQueue.drainTo(results);
        for (Writable result : results) {
            outputCollector.collect(new IntWritable(1), result);
        }
    }

    @Override
    public void configure(JobConf entries) {
        pool = Executors.newFixedThreadPool(20);
        List<Callable<Writable>> tasks = Lists.newArrayList();
        for (int i = 0; i < 20; i++) {
            tasks.add(new Callable<Writable>() {
                @Override
                public Writable call() throws Exception {
                    Writable in = inputQueue.poll();
                    while (in != null && in != endMarker) {
                        // do the job
                        Writable result = null;
                        outputQueue.add(result);
                        in = inputQueue.poll();
                    }
                    return null;
                }
            });
        }
        try {
            futures = pool.invokeAll(tasks);
        } catch (InterruptedException e) {
            // shouldn't happen
            throw new RuntimeException(e);
        }
    }
}
