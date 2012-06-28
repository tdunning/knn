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

package org.apache.mahout.knn.legacy;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.knn.generate.MultiNormal;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;

/**
 * Example of how to write a sequence file with vectors
 */
public class SampleSequenceFileWriter {
    public static List<Vector> writeTestFile(String fileName, int dimension, int records, boolean returnData) throws IOException {
        System.getProperties().setProperty("java.security.krb5.realm", "");
        System.getProperties().setProperty("java.security.krb5.kdc", "");
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.getLocal(conf);
        SequenceFile.Writer out = SequenceFile.createWriter(fs, conf, new Path(fileName), IntWritable.class, VectorWritable.class);
        MultiNormal s = new MultiNormal(dimension);
        VectorWritable vw = new VectorWritable();
        List<Vector> r = Lists.newArrayList();
        for (int i = 0; i < records; i++) {
            Vector v = s.sample();
            if (returnData) {
                r.add(v);
            }
            vw.set(v);
            out.append(new IntWritable(i), vw);
        }
        out.close();
        return r;
    }

    public static List<Vector> readTestFile(String filename) throws IOException {
        System.getProperties().setProperty("java.security.krb5.realm", "");
        System.getProperties().setProperty("java.security.krb5.kdc", "");
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.getLocal(conf);
        SequenceFile.Reader input = new SequenceFile.Reader(fs, new Path(filename), conf);
        IntWritable key = new IntWritable();
        List<Vector> r = Lists.newArrayList();
        while (input.next(key)) {
            VectorWritable vw = new VectorWritable();
            input.getCurrentValue(vw);
            r.add(vw.get());
        }
        return r;
    }
}
