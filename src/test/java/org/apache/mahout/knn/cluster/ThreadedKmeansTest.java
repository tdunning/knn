package org.apache.mahout.knn.cluster;

import com.google.common.collect.Lists;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.Searcher;
import org.apache.mahout.knn.UpdatableSearcher;
import org.apache.mahout.knn.WeightedVector;
import org.apache.mahout.knn.generate.MultiNormal;
import org.apache.mahout.knn.search.LocalitySensitiveHash;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.junit.Test;

import java.util.List;
import java.util.concurrent.ExecutionException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ThreadedKmeansTest {

    private static final int THREADS = 2;
    private static final int SPLITS = 2;

    @Test
    public void testClustering() throws ExecutionException, InterruptedException {

        // construct data samplers centered on the corners of a unit cube
        Matrix mean = new DenseMatrix(8, 3);
        List<MultiNormal> rowSamplers = Lists.newArrayList();
        for (int i = 0; i < 8; i++) {
            mean.viewRow(i).assign(new double[]{0.25 * (i & 4), 0.5 * (i & 2), i & 1});
            rowSamplers.add(new MultiNormal(0.01, mean.viewRow(i)));
        }

        // sample a bunch of data points
        Matrix data = new DenseMatrix(100000, 3);
        for (MatrixSlice row : data) {
            row.vector().assign(rowSamplers.get(row.index() % 8).sample());
        }

        // cluster the data
        final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();
        final int processors = Runtime.getRuntime().availableProcessors();
        System.out.printf("%d cores\n", processors);
        for (int i = 0; i < 4; i++) {
            for (int threads = 1; threads <= processors + 1; threads++) {
                for (int splits : new int[]{threads, 2 * threads}) {
                    clusterCheck(mean, "projection", data, new StreamingKmeans.SearchFactory() {
                        @Override
                        public UpdatableSearcher create() {
                            return new ProjectionSearch(3, distance, 4, 10);
                        }
                    }, threads, splits);
                    clusterCheck(mean, "lsh", data, new StreamingKmeans.SearchFactory() {
                        @Override
                        public UpdatableSearcher create() {
                            return new LocalitySensitiveHash(3, distance, 10);
                        }
                    }, threads, splits);

                }
            }
        }
    }

    private void clusterCheck(Matrix mean, String title, Matrix original, StreamingKmeans.SearchFactory searchFactory, int threads, int splits) throws ExecutionException, InterruptedException {
        List<Iterable<MatrixSlice>> data = Lists.newArrayList();
        int width = original.columnSize();
        int rowsPerSplit = (original.rowSize() + splits - 1) / splits;
        for (int i = 0; i < splits; i++) {
            final int rowOffset = i * rowsPerSplit;
            int rows = Math.min(original.rowSize() - rowOffset, rowsPerSplit);
            final Matrix split = original.viewPart(rowOffset, rows, 0, width).like();
            split.assign(original.viewPart(rowOffset, rows, 0, width));
            data.add(split);
        }

        long t0 = System.currentTimeMillis();
        Searcher r = new ThreadedKmeans().cluster(new EuclideanDistanceMeasure(), data, 1000, threads, searchFactory);
        long t1 = System.currentTimeMillis();

        assertEquals("Total weight not preserved", totalWeight(original), totalWeight(r), 1e-9);

        // and verify that each corner of the cube has a centroid very nearby
        for (MatrixSlice row : mean) {
            WeightedVector v = r.search(row.vector(), 1).get(0);
            assertTrue(v.getWeight() < 0.05);
        }
        System.out.printf("%s\t%d\t%d\t%.1f\n",
                title, threads, splits, (t1 - t0) / 1000.0 / original.rowSize() * 1e6);
    }

    private double totalWeight(Iterable<MatrixSlice> data) {
        double sum = 0;
        for (MatrixSlice row : data) {
            if (row.vector() instanceof WeightedVector) {
                sum += ((WeightedVector) row.vector()).getWeight();
            } else {
                sum++;
            }
        }
        return sum;
    }
}
