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

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Predicate;
import com.google.common.base.Splitter;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;
import com.google.common.io.Files;
import com.google.common.io.LineProcessor;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Read, tokenize and convert the 20 newsgroups test data to vector form.
 * <p/>
 * The vectorization is done using a hashed projection to a fixed dimension vector using a
 * selectable term weighting.
 *
 * Command line options are
 *
 * <ul>
 *   <li>weighting code, three characters long.  The first character can be l, s, t or x to indicate
 *   term weighting of log, square root, term frequency or no weighting.  The second
 *   character can be i or x to indicate IDF weighting or no corpus weighting.  The third character
 *   can be c or x to indicate cosine normalization or no normalization.</li>
 *   <li>a comma separated list of header lines to use</li>
 *   <li>a boolean to indicate whether quoted lines should be retained (true to retain, false to omit)</li>
 *   <li>the dimension of the result vector</li>
 *   <li>a list of directories containing files to parse</li>
 * </ul>
 */
public class Vectorize20NewsGroups {
  private static boolean includeQuotes;
  private static Set<String> legalHeaders;

  public static void main(String[] args) throws IOException {
    String weightingCode = args[0];
    CorpusWeighting cw = CorpusWeighting.parse(weightingCode);

    boolean normalize = weightingCode.endsWith("c");

    legalHeaders = Sets.newHashSet();
    Iterables.addAll(legalHeaders, Splitter.on(",").trimResults().split(args[1]));

    includeQuotes = Boolean.parseBoolean(args[2]);

    int dimension = Integer.parseInt(args[3]);

    SequenceFile.Writer sf = SequenceFile.createWriter(FileSystem.getLocal(null), new Configuration(), new Path("output"), String.class, VectorWritable.class);
    for (String file : Arrays.asList(args).subList(4, args.length - 4)) {
      recursivelyVectorize(sf, new File(file), cw, normalize, dimension);
    }
  }

  static void recursivelyVectorize(SequenceFile.Writer sf, File f, CorpusWeighting w, boolean normalize, int dimension) throws IOException {
    if (f.isDirectory()) {
      for (File file : f.listFiles()) {
        recursivelyVectorize(sf, file, null, false, dimension);
      }
    } else {
      Vector v = vectorizeFile(f, w, normalize, dimension);
      sf.append(f.getParentFile().getName(), new VectorWritable(v));
    }
  }

  static Vector vectorizeFile(File f, CorpusWeighting w, boolean normalize, int dimension) throws IOException {
    Multiset<String> counts = parse(f);
    return vectorize(counts, w, normalize, dimension);
  }

  static Vector vectorize(Multiset<String> doc, CorpusWeighting w, boolean normalize, int dimension) {
    Vector v = new RandomAccessSparseVector(dimension);
    FeatureVectorEncoder encoder = new StaticWordValueEncoder("text");
    for (String word : doc.elementSet()) {
      encoder.addToVector(word, w.weight(word, doc.count(word)), v);
    }
    if (normalize) {
      return v.normalize();
    } else {
      return v;
    }
  }

  static Multiset<String> parse(File f) throws IOException {
    return Files.readLines(f, Charsets.UTF_8, new LineProcessor<Multiset<String>>() {
      private boolean readingHeaders = true;
      private Splitter header = Splitter.on(":").limit(2);
      private Splitter words = Splitter.on(CharMatcher.forPredicate(new Predicate<Character>() {
        @Override
        public boolean apply(Character ch) {
          return !Character.isLetterOrDigit(ch) && ch != '.' && ch != '/' && ch != ':';
        }
      })).omitEmptyStrings().trimResults();

      private Pattern quotedLine = Pattern.compile("(^In article .*)|(^> .*)|(.*writes:$)|(^|>)");

      private Multiset<String> counts = HashMultiset.create();

      @Override
      public boolean processLine(String line) throws IOException {
        if (readingHeaders && line.length() == 0) {
          readingHeaders = false;
        }

        if (readingHeaders) {
          Iterator<String> i = header.split(line).iterator();
          String head = i.next();
          if (legalHeaders.contains(head)) {
            addText(counts, i.next());
          }
        } else {
          boolean quote = quotedLine.matcher(line).matches();
          if (includeQuotes || !quote) {
            addText(counts, line);
          }
        }
        return true;
      }

      @Override
      public Multiset<String> getResult() {
        return counts;
      }

      private void addText(Multiset<String> v, String line) {
        for (String word : words.split(line)) {
          v.add(word);
        }
      }
    });
  }

  private static abstract class CorpusWeighting {
    static Map<String, CorpusWeighting> corpusWeights = ImmutableMap.of("i", new Idf(), "x", new Unit());

    static CorpusWeighting parse(String code) {
      CorpusWeighting cw = corpusWeights.get(code.substring(1, 1));
      TermWeighting tw = TermWeighting.parse(code.substring(0, 1));
      cw.setTermWeighting(tw);
      return cw;
    }

    TermWeighting termWeighting;

    public void setTermWeighting(TermWeighting termWeighting) {
      this.termWeighting = termWeighting;
    }

    abstract double weight(String word, int count);
  }

  private static class Idf extends CorpusWeighting {
    Multiset documentFrequency;
    int corpusSize;

    @Override
    double weight(String word, int count) {
      return termWeighting.termFrequencyWeight(count) * Math.log((corpusSize + 1) / (documentFrequency.count(word) + 1));
    }
  }

  private static class Unit extends CorpusWeighting {
    @Override
    double weight(String word, int count) {
      return termWeighting.termFrequencyWeight(count);
    }
  }

  private static abstract class TermWeighting {
    abstract double termFrequencyWeight(int count);

    static final TermWeighting log = new TermWeighting() {
      @Override
      double termFrequencyWeight(int count) {
        return Math.log(count);
      }
    };
    static final TermWeighting linear = new TermWeighting() {
      @Override
      double termFrequencyWeight(int count) {
        return count;
      }
    };
    static final TermWeighting root = new TermWeighting() {
      @Override
      double termFrequencyWeight(int count) {
        return Math.sqrt(count);
      }
    };
    static final TermWeighting unit = new TermWeighting() {
      @Override
      double termFrequencyWeight(int count) {
        return 1;
      }
    };

    static Map<String, TermWeighting> termWeights = ImmutableMap.of("l", TermWeighting.log, "s", TermWeighting.root, "t", TermWeighting.linear, "x", TermWeighting.unit);

    static final TermWeighting parse(String code) {
      return termWeights.get(code);
    }


  }

}
