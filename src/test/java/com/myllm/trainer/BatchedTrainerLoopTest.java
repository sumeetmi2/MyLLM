package com.myllm.trainer;

import static com.myllm.utils.TestUtils.assertMatrixEquals;
import static org.junit.jupiter.api.Assertions.*;

import com.myllm.batched.Batch;
import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.dao.Matrix;
import com.myllm.optimizer.AdamOptimizer;
import com.myllm.transformer.TransformerModel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class BatchedTrainerLoopTest {

    private static final int VOCAB = 7;
    private static final int T = 6;
    private static final double TOLERANCE = 1e-5;

    private static TransformerModel newModel() {
        return new TransformerModel(VOCAB, T, 8, 2, 16, 2, new Random(42));
    }

    private static int[] corpus() {
        return new Random(5).ints(200, 0, VOCAB).toArray();
    }

    private static BatchedTrainerLoop batchedTrainer(TransformerModel model, int batch, int accum) {
        return new BatchedTrainerLoop(
                model,
                new CrossEntropyLoss(),
                new AdamOptimizer(0.01),
                corpus(),
                T,
                batch,
                accum,
                new Random(42),
                null);
    }

    // sequences with masks of different counts, so the token weighting matters
    private static List<int[]> inputs() {
        Random r = new Random(9);
        return List.of(
                r.ints(6, 0, VOCAB).toArray(),
                r.ints(4, 0, VOCAB).toArray(),
                r.ints(5, 0, VOCAB).toArray(),
                r.ints(6, 0, VOCAB).toArray());
    }

    private static List<int[]> targets() {
        Random r = new Random(10);
        return List.of(
                r.ints(6, 0, VOCAB).toArray(),
                r.ints(4, 0, VOCAB).toArray(),
                r.ints(5, 0, VOCAB).toArray(),
                r.ints(6, 0, VOCAB).toArray());
    }

    private static List<boolean[]> masks() {
        return Arrays.asList(
                new boolean[] {false, false, true, true, true, true},
                null,
                new boolean[] {false, false, false, false, true},
                new boolean[] {true, false, true, false, true, false});
    }

    // splits the four sequences into micro-batches of `size`
    private static List<Batch> split(int size) {
        List<Batch> batches = new ArrayList<>();
        for (int from = 0; from < 4; from += size) {
            batches.add(Batch.stack(
                    inputs().subList(from, from + size),
                    targets().subList(from, from + size),
                    masks().subList(from, from + size)));
        }
        return batches;
    }

    @Test
    @DisplayName("Accumulation gives the same gradient and loss however the sequences are split into micro-batches")
    void accumulationMatchesOneBatchTest() {
        BatchedTrainerLoop.StepGradients one = batchedTrainer(newModel(), 4, 1).accumulate(split(4));
        for (int size : new int[] {2, 1}) {
            BatchedTrainerLoop.StepGradients split =
                    batchedTrainer(newModel(), size, 4 / size).accumulate(split(size));
            assertEquals(one.loss(), split.loss(), TOLERANCE, "loss, micro-batch size " + size);
            for (int p = 0; p < one.gradients().size(); p++) {
                assertMatrixEquals(
                        one.gradients().get(p).toArray(), split.gradients().get(p), TOLERANCE);
            }
        }
    }

    @Test
    @DisplayName("Step gradient is the token-weighted mean of each sequence's own mean-loss gradient")
    void tokenWeightedMeanTest() {
        TransformerModel model = newModel();
        BatchedTrainerLoop.StepGradients batched = batchedTrainer(model, 4, 1).accumulate(split(4));

        // reference: simple model, one sequence at a time, weighted by counted tokens
        CrossEntropyLoss loss = new CrossEntropyLoss();
        List<Matrix> expected = null;
        double lossSum = 0;
        int total = 0;
        for (int s = 0; s < 4; s++) {
            boolean[] mask = masks().get(s);
            int counted = mask == null ? inputs().get(s).length : (int) countTrue(mask);
            lossSum += loss.forwardPass(
                            model.forwardPass(inputs().get(s)), targets().get(s), mask)
                    * counted;
            List<Matrix> g =
                    model.backwardPass(loss.backwardPass().scale(counted)).trainableGradients();
            if (expected == null) {
                expected = g;
            } else {
                for (int p = 0; p < g.size(); p++) {
                    expected.set(p, expected.get(p).add(g.get(p)));
                }
            }
            total += counted;
        }
        assertEquals(lossSum / total, batched.loss(), TOLERANCE);
        for (int p = 0; p < expected.size(); p++) {
            assertMatrixEquals(
                    expected.get(p).scale(1.0 / total).toArray(),
                    batched.gradients().get(p),
                    TOLERANCE);
        }
    }

    private static long countTrue(boolean[] mask) {
        long n = 0;
        for (boolean m : mask) {
            if (m) {
                n++;
            }
        }
        return n;
    }

    @Test
    @DisplayName("batch=1 accum=1 trains exactly like the simple TrainerLoop")
    void singleSequenceMatchesTrainerLoopTest() {
        TransformerModel simpleModel = newModel();
        TransformerModel batchedModel = newModel();
        new TrainerLoop(simpleModel, new CrossEntropyLoss(), new AdamOptimizer(0.01), corpus(), T, new Random(42))
                .train(5, 5);
        batchedTrainer(batchedModel, 1, 1).train(5, 5);
        assertEquals(simpleModel.parameters(), batchedModel.parameters());
    }

    @Test
    @DisplayName("Loss goes down with batched training")
    void lossDecreasesTest() {
        List<Double> losses = batchedTrainer(newModel(), 4, 2).train(60, 20);
        assertTrue(losses.get(2) < losses.get(0), losses.toString());
    }

    @Test
    @DisplayName("batch and accum below 1 are rejected")
    void rejectsBadSettingsTest() {
        assertThrows(IllegalArgumentException.class, () -> batchedTrainer(newModel(), 0, 1));
        assertThrows(IllegalArgumentException.class, () -> batchedTrainer(newModel(), 1, 0));
    }
}
