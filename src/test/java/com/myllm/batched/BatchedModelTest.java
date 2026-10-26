package com.myllm.batched;

import static com.myllm.utils.TestUtils.assertMatrixEquals;
import static com.myllm.utils.TestUtils.randomMatrix;
import static org.junit.jupiter.api.Assertions.*;

import com.myllm.dao.Matrix;
import com.myllm.transformer.TransformerModel;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

// the simple TransformerModel is the reference: the batched route must give the same logits and gradients
public class BatchedModelTest {

    private static final int VOCAB = 11;
    private static final int T_MAX = 16;
    private static final int D_MODEL = 8;
    private static final int HEADS = 2;
    private static final int D_FF = 16;
    private static final int BLOCKS = 2;
    private static final double TOLERANCE = 1e-4;

    private static TransformerModel newModel() {
        return new TransformerModel(VOCAB, T_MAX, D_MODEL, HEADS, D_FF, BLOCKS, new Random(42));
    }

    private static int[] randomIds(Random r, int length) {
        return r.ints(length, 0, VOCAB).toArray();
    }

    private static int[] concat(int[]... parts) {
        return Arrays.stream(parts).flatMapToInt(Arrays::stream).toArray();
    }

    private static Matrix rows(Matrix m, int from, int count) {
        return BatchOps.rows(m, from, count);
    }

    @Test
    @DisplayName("One sequence: batched logits and gradients equal the simple model's exactly")
    void singleSequenceMatchesExactlyTest() {
        assertSingleSequenceExact(newModel());
    }

    @Test
    @DisplayName("One sequence with LoRA on: batched logits and gradients equal the simple model's exactly")
    void singleSequenceWithLoraTest() {
        TransformerModel model = newModel();
        model.enableLora(2, 4.0, true, new Random(7));
        // B starts at zero, which would hide a wrong adapter path; give it values
        for (Matrix p : model.loraParameters()) {
            com.myllm.utils.TestUtils.fillRandom(new Random(3), p, 0.1);
        }
        assertSingleSequenceExact(model);
    }

    private static void assertSingleSequenceExact(TransformerModel model) {
        Random r = new Random(1);
        int[] ids = randomIds(r, 12);
        Matrix dLogits = randomMatrix(r, ids.length, VOCAB);

        Matrix simpleLogits = model.forwardPass(ids);
        List<Matrix> simpleGrads = model.backwardPass(dLogits).trainableGradients();

        BatchedModel batched = new BatchedModel(model);
        Matrix batchedLogits = batched.forwardPass(ids, new int[] {ids.length});
        List<Matrix> batchedGrads = batched.backwardPass(dLogits).trainableGradients();

        assertEquals(simpleLogits, batchedLogits);
        assertEquals(simpleGrads.size(), batchedGrads.size());
        for (int p = 0; p < simpleGrads.size(); p++) {
            assertEquals(simpleGrads.get(p), batchedGrads.get(p), "gradient " + p);
        }
    }

    @Test
    @DisplayName(
            "Stacked sequences of different lengths give each sequence's own logits (no cross-attention, positions restart)")
    void segmentsAreIndependentTest() {
        TransformerModel model = newModel();
        Random r = new Random(2);
        int[] a = randomIds(r, 5);
        int[] b = randomIds(r, 9);
        int[] c = randomIds(r, 16);
        Matrix logits = new BatchedModel(model).forwardPass(concat(a, b, c), new int[] {5, 9, 16});

        assertMatrixEquals(model.forwardPass(a).toArray(), rows(logits, 0, 5), TOLERANCE);
        assertMatrixEquals(model.forwardPass(b).toArray(), rows(logits, 5, 9), TOLERANCE);
        assertMatrixEquals(model.forwardPass(c).toArray(), rows(logits, 14, 16), TOLERANCE);
    }

    @Test
    @DisplayName("Batched gradients are the sum of each sequence's gradients")
    void gradientsSumOverSequencesTest() {
        assertGradientsSum(newModel());
    }

    @Test
    @DisplayName("Batched LoRA gradients are the sum of each sequence's gradients")
    void loraGradientsSumOverSequencesTest() {
        TransformerModel model = newModel();
        model.enableLora(2, 4.0, true, new Random(7));
        for (Matrix p : model.loraParameters()) {
            com.myllm.utils.TestUtils.fillRandom(new Random(3), p, 0.1);
        }
        assertGradientsSum(model);
    }

    private static void assertGradientsSum(TransformerModel model) {
        Random r = new Random(3);
        int[] a = randomIds(r, 6);
        int[] b = randomIds(r, 10);
        Matrix dA = randomMatrix(r, 6, VOCAB);
        Matrix dB = randomMatrix(r, 10, VOCAB);
        Matrix dBoth = new Matrix(16, VOCAB);
        System.arraycopy(dA.data(), 0, dBoth.data(), 0, dA.data().length);
        System.arraycopy(dB.data(), 0, dBoth.data(), dA.data().length, dB.data().length);

        model.forwardPass(a);
        List<Matrix> gA = model.backwardPass(dA).trainableGradients();
        model.forwardPass(b);
        List<Matrix> gB = model.backwardPass(dB).trainableGradients();

        BatchedModel batched = new BatchedModel(model);
        batched.forwardPass(concat(a, b), new int[] {6, 10});
        List<Matrix> gBoth = batched.backwardPass(dBoth).trainableGradients();
        for (int p = 0; p < gBoth.size(); p++) {
            assertMatrixEquals(gA.get(p).add(gB.get(p)).toArray(), gBoth.get(p), TOLERANCE);
        }
    }

    @Test
    @DisplayName("Bad lengths are rejected")
    void rejectsBadLengthsTest() {
        BatchedModel batched = new BatchedModel(newModel());
        int[] ids = new int[20];
        assertThrows(IllegalArgumentException.class, () -> batched.forwardPass(ids, new int[] {10, 0, 10}));
        assertThrows(
                IllegalArgumentException.class, () -> batched.forwardPass(ids, new int[] {2, 17}), "longer than T_max");
        assertThrows(IllegalArgumentException.class, () -> batched.forwardPass(ids, new int[] {10, 9}), "sum differs");
    }

    @Test
    @DisplayName("Batch.stack concatenates sequences and merges masks; null masks count every position")
    void stackTest() {
        Batch batch = Batch.stack(
                List.of(new int[] {1, 2}, new int[] {3, 4, 5}),
                List.of(new int[] {2, 3}, new int[] {4, 5, 6}),
                Arrays.asList(new boolean[] {false, true}, null));
        assertArrayEquals(new int[] {1, 2, 3, 4, 5}, batch.ids());
        assertArrayEquals(new int[] {2, 3}, batch.lengths());
        assertArrayEquals(new int[] {2, 3, 4, 5, 6}, batch.targets());
        assertArrayEquals(new boolean[] {false, true, true, true, true}, batch.mask());
        assertEquals(4, batch.countedTokens());
        assertNull(
                Batch.stack(List.of(new int[] {1}), List.of(new int[] {2}), Collections.singletonList((boolean[]) null))
                        .mask());
    }
}
