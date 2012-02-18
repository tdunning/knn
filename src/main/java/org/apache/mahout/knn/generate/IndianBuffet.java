package org.apache.mahout.knn.generate;

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.LineProcessor;
import com.google.common.io.Resources;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Samples a "document" from an IndianBuffet process.
 *
 * See http://mlg.eng.cam.ac.uk/zoubin/talks/turin09.pdf for details
 */
public class IndianBuffet<T> implements Sampler<List<T>> {
  private List<Integer> count = Lists.newArrayList();
  private int documents = 0;
  private double alpha;
  private WordFunction<T> converter = null;
  private Random gen = new Random();

  public IndianBuffet(double alpha, WordFunction<T> converter) {
    this.alpha = alpha;
    this.converter = converter;
  }
  
  public static IndianBuffet<Integer> createIntegerDocumentSampler(double alpha) {
    return new IndianBuffet<Integer>(alpha, new IdentityConverter());
  }

  public static IndianBuffet<String> createTextDocumentSampler(double alpha) {
    return new IndianBuffet<String>(alpha, new WordConverter());
  }

  public List<T> sample() {
    List<T> r = Lists.newArrayList();
    if (documents == 0) {
      int n = new PoissonSampler(alpha).sample();
      for (int i = 0; i < n; i++) {
        r.add(converter.convert(i));
        count.add(1);
      }
      documents++;
    } else {
      documents++;
      int i = 0;
      for (double cnt : count) {
        if (gen.nextDouble() < cnt / documents) {
          r.add(converter.convert(i));
          count.set(i, count.get(i) + 1);
        }
        i++;
      }
      final int newItems = new PoissonSampler(alpha / documents).sample();
      for (int j = 0; j < newItems; j++) {
        r.add(converter.convert(i + j));
        count.add(1);
      }
    }
    return r;
  }

  private interface WordFunction<T> {
    T convert(int i);
  }

  /**
   * Just converts to an integer.
   */
  public static class IdentityConverter implements WordFunction<Integer> {
    public Integer convert(int i) {
      return i;
    }
  }

  /**
   * Converts to a string.
   */
  public static class StringConverter implements WordFunction<String> {
    public String convert(int i) {
      return "" + i;
    }
  }

  /**
   * Converts to one of a list of common English words for reasonably small integers and converts
   * to a token like w_92463 for big integers.
   */
  public static class WordConverter implements WordFunction<String> {
    private Splitter onSpace = Splitter.on(CharMatcher.WHITESPACE).omitEmptyStrings().trimResults();
    private List<String> words;

    public WordConverter() {
      try {
        words = Resources.readLines(Resources.getResource("words.txt"), Charsets.UTF_8, new LineProcessor<List<String>>() {
          List<String> words = Lists.newArrayList();

          public boolean processLine(String line) throws IOException {
            Iterables.addAll(words, onSpace.split(line));
            return true;
          }

          public List<String> getResult() {
            return words;
          }
        });
      } catch (IOException e) {
        throw new ImpossibleException(e);
      }
    }

    public String convert(int i) {
      if (i < words.size()) {
        return words.get(i);
      } else {
        return "w_" + i;
      }
    }
  }

  public static class ImpossibleException extends RuntimeException {
    public ImpossibleException(Throwable e) {
      super(e);
    }
  }
}
