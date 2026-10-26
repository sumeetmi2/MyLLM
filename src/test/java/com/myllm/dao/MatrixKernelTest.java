package com.myllm.dao;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

// checks the SIMD products against a plain triple loop in double, on shapes that hit every
// vector / tail combination: n below the lane count, c not a multiple of 4, single rows and columns
public class MatrixKernelTest {

    // r, n, c for the product (r x n) * (n x c)
    static Stream<int[]> shapes() {
        return Stream.of(
                new int[] {1, 1, 1},
                new int[] {3, 5, 7},
                new int[] {17, 33, 9},
                new int[] {4, 4, 4},
                new int[] {2, 3, 6},
                new int[] {64, 128, 32},
                new int[] {5, 16, 1});
    }

    private static Matrix random(Random rnd, int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        for (int k = 0; k < m.data().length; k++) {
            m.data()[k] = (float) rnd.nextGaussian();
        }
        return m;
    }

    // reference A * B in double from plain get() calls
    private static double[][] reference(Matrix a, Matrix b) {
        double[][] out = new double[a.rows()][b.cols()];
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < b.cols(); j++) {
                double sum = 0;
                for (int k = 0; k < a.cols(); k++) {
                    sum += (double) a.get(i, k) * b.get(k, j);
                }
                out[i][j] = sum;
            }
        }
        return out;
    }

    // float sums carry ~1e-7 relative error per add; scale the allowance with the dot length and magnitude
    private static void assertClose(double[][] expected, Matrix actual, int n) {
        assertEquals(expected.length, actual.rows(), "rows");
        assertEquals(expected[0].length, actual.cols(), "cols");
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                double delta = 1e-5 * Math.max(1.0, Math.abs(expected[i][j])) * Math.sqrt(n);
                assertEquals(expected[i][j], actual.get(i, j), delta, "[" + i + "][" + j + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("shapes")
    @DisplayName("multiply matches a plain triple loop")
    void multiplyTest(int[] s) {
        Random rnd = new Random(42);
        Matrix a = random(rnd, s[0], s[1]);
        Matrix b = random(rnd, s[1], s[2]);
        assertClose(reference(a, b), a.multiply(b), s[1]);
    }

    @ParameterizedTest
    @MethodSource("shapes")
    @DisplayName("multiplyTranspose matches multiplying by the explicit transpose")
    void multiplyTransposeTest(int[] s) {
        Random rnd = new Random(42);
        Matrix a = random(rnd, s[0], s[1]);
        Matrix bT = random(rnd, s[2], s[1]);
        assertClose(reference(a, bT.transpose()), a.multiplyTranspose(bT), s[1]);
    }

    @ParameterizedTest
    @MethodSource("shapes")
    @DisplayName("transposeMultiply matches multiplying the explicit transpose")
    void transposeMultiplyTest(int[] s) {
        Random rnd = new Random(42);
        Matrix aT = random(rnd, s[1], s[0]);
        Matrix b = random(rnd, s[1], s[2]);
        assertClose(reference(aT.transpose(), b), aT.transposeMultiply(b), s[1]);
    }

    @Test
    @DisplayName("mixed large and small values stay accurate")
    void mixedMagnitudesTest() {
        Random rnd = new Random(7);
        Matrix a = random(rnd, 9, 37);
        Matrix b = random(rnd, 37, 11);
        for (int k = 0; k < a.data().length; k += 5) {
            a.data()[k] *= 1000f;
        }
        for (int k = 0; k < b.data().length; k += 7) {
            b.data()[k] *= 1e-3f;
        }
        assertClose(reference(a, b), a.multiply(b), 37);
    }
}
