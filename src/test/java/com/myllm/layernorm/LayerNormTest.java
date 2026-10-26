package com.myllm.layernorm;

import static com.myllm.utils.TestUtils.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.myllm.dao.Matrix;
import java.util.Random;
import java.util.function.DoubleSupplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class LayerNormTest {

    @Test
    @DisplayName("LayerNorm forward normalises each row")
    void forwardPassTest() {
        LayerNorm layerNorm = new LayerNorm(3);
        Matrix y = layerNorm.forwardPass(Matrix.of(new double[][] {{1, 2, 3}, {4, 4, 4}}));
        double v = 1.0 / Math.sqrt(2.0 / 3.0 + 1e-5);
        assertMatrixEquals(new double[][] {{-v, 0, v}, {0, 0, 0}}, y, 1e-6);
    }

    @Test
    @DisplayName("LayerNorm backward: dX rows sum to zero")
    void backwardRowSumTest() {
        Random random = new Random(42);
        LayerNorm layerNorm = new LayerNorm(3);
        layerNorm.forwardPass(randomMatrix(random, 2, 3));
        Matrix dX = layerNorm.backwardPass(randomMatrix(random, 2, 3)).dX();
        for (double[] row : dX.toArray()) {
            double sum = 0;
            for (double d : row) {
                sum += d;
            }
            assertEquals(0.0, sum, 1e-6);
        }
    }

    @Test
    @DisplayName("LayerNorm backward matches numerical gradient")
    void backwardPassTest() {
        Random random = new Random(42);
        LayerNorm layerNorm = new LayerNorm(3);
        // default gamma=1, beta=0 can hide a missing gamma in dXhat = dY * gamma, so randomise the layer's own params
        fillRandom(random, layerNorm.getGamma(), 1.0);
        fillRandom(random, layerNorm.getBeta(), 1.0);
        Matrix x = randomMatrix(random, 2, 3);
        Matrix dY = randomMatrix(random, 2, 3);

        layerNorm.forwardPass(x);
        LayerNorm.LayerNormBackwardResult result = layerNorm.backwardPass(dY);

        DoubleSupplier loss = weightedSumLoss(() -> layerNorm.forwardPass(x), dY);

        assertGradient("dX", x, result.dX(), loss);
        assertGradient("dGamma", layerNorm.getGamma(), result.dGamma(), loss);
        assertGradient("dBeta", layerNorm.getBeta(), result.dBeta(), loss);
    }
}
