package com.myllm.attention;

import static com.myllm.utils.TestUtils.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.myllm.dao.Matrix;
import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleSupplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class MultiHeadAttentionTest {

    // T=3, d_model=4, h=2 -> d_k=2. T != d_model so row/col mix-ups show up
    private static final int T = 3;
    private static final int D_MODEL = 4;
    private static final int HEADS = 2;

    @Test
    @DisplayName("MHA forward output keeps input shape (T x d_model)")
    void forwardPassShapeTest() {
        MultiHeadAttention mha = new MultiHeadAttention(D_MODEL, HEADS, new Random(42), 0.02, 0.02);
        Matrix output = mha.forwardPass(randomMatrix(new Random(42), T, D_MODEL));
        assertShape(T, D_MODEL, output);
    }

    @Test
    @DisplayName("MHA rejects d_model not divisible by number of heads")
    void invalidHeadCountTest() {
        assertThrows(IllegalArgumentException.class, () -> new MultiHeadAttention(4, 3, new Random(42), 0.02, 0.02));
    }

    @Test
    @DisplayName("MHA forward is causal: changing a future token does not change earlier outputs")
    void forwardPassCausalTest() {
        Random r = new Random(42);
        MultiHeadAttention mha = new MultiHeadAttention(D_MODEL, HEADS, new Random(42), 0.02, 0.02);
        Matrix x = randomMatrix(r, T, D_MODEL);
        Matrix before = mha.forwardPass(x);

        // change only the last token
        for (int j = 0; j < D_MODEL; j++) {
            x.set(T - 1, j, x.get(T - 1, j) + 5.0f);
        }
        Matrix after = mha.forwardPass(x);

        // rows 0..T-2 only attend to positions <= themselves, so they must be unchanged
        assertMatrixEquals(
                Arrays.copyOf(before.toArray(), T - 1), Matrix.of(Arrays.copyOf(after.toArray(), T - 1)), 1e-12);
    }

    @Test
    @DisplayName("MHA backward matches numerical gradient")
    void backwardPassTest() {
        Random r = new Random(42);
        MultiHeadAttention mha = new MultiHeadAttention(D_MODEL, HEADS, new Random(42), 0.02, 0.02);
        // default init [-0.1, 0.1] gives near-uniform softmax, which hides softmax-backward bugs.
        // larger weights -> peaked attention, exercises the full softmax jacobian
        fillRandom(r, mha.getWO(), 0.5);
        for (AttentionHead head : mha.getAttentionHeads()) {
            fillRandom(r, head.getWQ(), 0.5);
            fillRandom(r, head.getWK(), 0.5);
            fillRandom(r, head.getWV(), 0.5);
        }
        Matrix x = randomMatrix(r, T, D_MODEL);
        Matrix dY = randomMatrix(r, T, D_MODEL);

        mha.forwardPass(x);
        MultiHeadAttention.MHABackwardResult result = mha.backwardPass(dY);
        DoubleSupplier loss = weightedSumLoss(() -> mha.forwardPass(x), dY);

        assertGradient("dX", x, result.dX(), loss);
        assertGradient("dWo", mha.getWO(), result.dWo(), loss);
        for (int h = 0; h < HEADS; h++) {
            AttentionHead head = mha.getAttentionHeads()[h];
            AttentionHead.AttentionBackwardResult headResult =
                    result.perHeadGradients().get(h);
            assertGradient("head" + h + ".dWq", head.getWQ(), headResult.dWq(), loss);
            assertGradient("head" + h + ".dWk", head.getWK(), headResult.dWk(), loss);
            assertGradient("head" + h + ".dWv", head.getWV(), headResult.dWv(), loss);
        }
    }
}
