package org.apache.mahout.knn.generate;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import org.junit.Test;

import java.util.List;

public class IndianBuffetTest {
  @Test
  public void testBasicText() {
    IndianBuffet<String> sampler = IndianBuffet.createTextDocumentSampler(30);
    Multiset<String> counts = HashMultiset.create();
    int[] lengths = new int[100];
    for (int i = 0; i < 30; i++) {
      final List<String> doc = sampler.sample();
      lengths[doc.size()]++;
      for (String w : doc) {
        counts.add(w);
      }
      System.out.printf("%s\n", doc);
    }
  }
}
