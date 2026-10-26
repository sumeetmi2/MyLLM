package com.myllm.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.myllm.dao.Matrix;
import com.myllm.train.GradCheck;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.Supplier;

public class TestUtils {
    // weights are float, so finite differences are only good to ~1e-3 relative (see GradCheck)
    private static final double TOLERANCE = 1e-2;

    public static Matrix randomMatrix(Random random, int rows, int cols) {
        Matrix result = new Matrix(rows, cols);
        float[] data = result.data();
        for (int k = 0; k < data.length; k++) {
            data[k] = (float) random.nextGaussian();
        }
        return result;
    }

    // overwrite an existing (e.g. layer-owned) matrix in place with gaussian * scale
    public static void fillRandom(Random random, Matrix target, double scale) {
        float[] data = target.data();
        for (int k = 0; k < data.length; k++) {
            data[k] = (float) (random.nextGaussian() * scale);
        }
    }

    public static void setValues(Matrix target, double[][] values) {
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                target.set(i, j, (float) values[i][j]);
            }
        }
    }

    public static void assertMatrixEquals(double[][] expected, Matrix actual, double delta) {
        assertShape(expected.length, expected[0].length, actual);
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                assertEquals(expected[i][j], actual.get(i, j), delta, "[" + i + "][" + j + "]");
            }
        }
    }

    public static void assertShape(int rows, int cols, Matrix actual) {
        assertEquals(rows, actual.rows(), "rows");
        assertEquals(cols, actual.cols(), "cols");
    }

    // scalar loss for single-layer grad checks: L = sum(forward() ⊙ dY), so dL/dOutput = dY
    public static DoubleSupplier weightedSumLoss(Supplier<Matrix> forward, Matrix dY) {
        return () -> GradCheck.weightedSum(forward.get(), dY);
    }

    public static void assertGradient(GradCheck.GradCheckResult result) {
        assertTrue(result.passed(TOLERANCE), result.toString());
    }

    // shape check first so a mismatched param/gradient pair fails loudly instead of as relErr=1.0
    public static void assertGradient(String name, Matrix param, Matrix analytic, DoubleSupplier loss) {
        assertEquals(
                param.rows() + "x" + param.cols(),
                analytic.rows() + "x" + analytic.cols(),
                name + ": param and analytic gradient shapes differ");
        assertGradient(GradCheck.check(name, param, analytic, loss));
    }
}
