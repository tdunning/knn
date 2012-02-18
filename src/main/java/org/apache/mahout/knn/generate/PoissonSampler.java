package org.apache.mahout.knn.generate;

import com.google.common.collect.Lists;
import org.apache.commons.math.distribution.PoissonDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;

import java.util.List;
import java.util.Random;

/**
 * Samples from a Poisson distribution.  Should probably not be used for lambda > 1000 or so.
 */
public class PoissonSampler implements Sampler<Integer> {
  double limit = 1;

  private Multinomial<Integer> partial;
  private Random gen = new Random();
  private PoissonDistribution pd;

  public PoissonSampler(double lambda) {
    pd = new PoissonDistributionImpl(lambda);
  }

  public Integer sample() {
    return sample(gen.nextDouble());
  }

  public Integer sample(double u) {
    if (u < limit) {
      List<WeightedThing<Integer>> steps = Lists.newArrayList();
      limit = 1;
      for (int i = 0; u / 20 < limit; i++) {
        final double pdf = pd.probability(i);
        limit -= pdf;
        steps.add(new WeightedThing<Integer>(i, pdf));
      }
      steps.add(new WeightedThing<Integer>(steps.size(), limit));
      partial = new Multinomial<Integer>(20, steps);
    }
    return partial.sample(u);
  }
}
