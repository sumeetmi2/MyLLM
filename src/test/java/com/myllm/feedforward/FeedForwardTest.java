package com.myllm.feedforward;

import static com.myllm.utils.TestUtils.*;

import com.myllm.dao.Matrix;
import java.util.Random;
import java.util.function.DoubleSupplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class FeedForwardTest {

    @Test
    @DisplayName("FeedForward forwardpass output keeps input shape (T x d_model)")
    void forwardPassShapeTest() {
        Random r = new Random(42);
        FeedForward ff = new FeedForward(3, 5, new Random(42), 0.02, 0.02);
        // the check nudges weights by 1e-2 (float needs a big step); with 0.02 init, pre-activations sit within that
        // of the ReLU kink and the finite difference jumps across it. Larger weights keep them clear of 0
        fillRandom(r, ff.getW_1(), 1.0);
        fillRandom(r, ff.getW_2(), 1.0);
        Matrix input = randomMatrix(r, 2, 3);
        Matrix result = ff.forwardPass(input);
        assertShape(2, 3, result);
    }

    @Test
    @DisplayName("FeedForward forwardpass enables activation of neuron")
    void forwardPassTest() {
        FeedForward ff = new FeedForward(2, 3, new Random(42), 0.02, 0.02);
        // set known weights in place: W_1 (2x3), B_1 (1x3), W_2 (3x2), B_2 (1x2)
        setValues(ff.getW_1(), new double[][] {{1, -1, 0.5}, {2, 1, -1}});
        setValues(ff.getB_1(), new double[][] {{0.5, 0, -1}});
        setValues(ff.getW_2(), new double[][] {{1, 0}, {0, 1}, {1, 1}});
        setValues(ff.getB_2(), new double[][] {{0.1, -0.1}});

        // row 0: H = [3.5, 0, -1.5]  -> relu [3.5, 0, 0] -> Y = [3.5, 0] + B_2 = [3.6, -0.1]
        // row 1: H = [-0.5, 1, -1.5] -> relu [0, 1, 0]   -> Y = [0, 1]   + B_2 = [0.1, 0.9]
        // without relu row 0 would be [2.1, -1.6], so this also proves negative units are switched off
        Matrix input = Matrix.of(new double[][] {{1, 1}, {-1, 0}});
        Matrix result = ff.forwardPass(input);

        assertMatrixEquals(new double[][] {{3.6, -0.1}, {0.1, 0.9}}, result, 1e-6);
    }

    @Test
    @DisplayName("Backwardpass check gradient")
    void backwardPassTest() {
        Random r = new Random(42);
        FeedForward ff = new FeedForward(3, 5, new Random(42), 0.02, 0.02);
        // the check nudges weights by 1e-2 (float needs a big step); with 0.02 init, pre-activations sit within that
        // of the ReLU kink and the finite difference jumps across it. Larger weights keep them clear of 0
        fillRandom(r, ff.getW_1(), 1.0);
        fillRandom(r, ff.getW_2(), 1.0);
        Matrix input = randomMatrix(r, 2, 3);
        Matrix dY = randomMatrix(r, 2, 3);
        ff.forwardPass(input);
        FeedForward.FeedForwardBackwardPassResult result = ff.backwardPass(dY);
        DoubleSupplier loss = weightedSumLoss(() -> ff.forwardPass(input), dY);
        assertGradient("d_X", input, result.d_X(), loss);
        assertGradient("d_W_1", ff.getW_1(), result.d_W_1(), loss);
        assertGradient("d_W_2", ff.getW_2(), result.d_W_2(), loss);
        assertGradient("d_B_1", ff.getB_1(), result.d_B_1(), loss);
        assertGradient("d_B_2", ff.getB_2(), result.d_B_2(), loss);
    }
}
