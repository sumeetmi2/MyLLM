package com.myllm.dao;

import java.util.Arrays;
import java.util.Random;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * A {@code rows x cols} matrix stored row-major in one flat float array: element {@code (i, j)} is
 * {@code data[i * cols + j]}.
 *
 * <p>Float halves memory and doubles the values per SIMD register; long sums (softmax, column sums) still add in
 * double. One array instead of an array per row means one bounds check and pointer hop less per access, and rows
 * sit next to each other in memory, which the hot loops rely on by walking a row start offset.
 */
public record Matrix(int rows, int cols, float[] data) {
    // below this much work (multiply-adds or elements) a parallel loop costs more in task overhead than it saves
    private static final long PARALLEL_THRESHOLD = 32_768;
    // widest vector this CPU supports: 4 floats on Apple NEON, 8 with AVX2, 16 with AVX-512
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public Matrix {
        if (rows <= 0 || cols <= 0 || data.length != rows * cols) {
            throw new IllegalArgumentException(
                    "can't make a " + rows + "x" + cols + " matrix from " + data.length + " values");
        }
    }

    public Matrix(int rows, int cols) {
        this(rows, cols, new float[rows * cols]);
    }

    // copies a nested array, rounding to float; for tests and fixtures, not hot paths
    public static Matrix of(double[][] values) {
        int rows = values.length;
        int cols = values[0].length;
        float[] data = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            if (values[i].length != cols) {
                throw new IllegalArgumentException(
                        "row " + i + " has " + values[i].length + " values, expected " + cols);
            }
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = (float) values[i][j];
            }
        }
        return new Matrix(rows, cols, data);
    }

    // widened to double so tests can compare against double fixtures
    public double[][] toArray() {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i * cols + j];
            }
        }
        return result;
    }

    public float get(int i, int j) {
        return data[i * cols + j];
    }

    public void set(int i, int j, float value) {
        data[i * cols + j] = value;
    }

    // index in data where row i starts
    public int offset(int i) {
        return i * cols;
    }

    // runs body(i) for every row i, in parallel when the op is big enough. Each row must only write its own output row
    private static void forEachRow(int rows, long work, IntConsumer body) {
        if (work >= PARALLEL_THRESHOLD) {
            IntStream.range(0, rows).parallel().forEach(body);
        } else {
            for (int i = 0; i < rows; i++) {
                body.accept(i);
            }
        }
    }

    private long size() {
        return (long) rows * cols;
    }

    // all three products go through one SIMD kernel that computes A * B^T (every output is a row-by-row dot product).
    // the other two transpose their inputs first; a transpose only moves n x c values against r x n x c multiply-adds
    public Matrix multiply(Matrix other) {
        if (cols != other.rows) {
            throw new IllegalArgumentException("Metrics Multiplication cannot be done");
        }
        return dotKernel(this, other.transpose());
    }

    // this^T * other: (n x r)^T * (n x c) = r x c
    public Matrix transposeMultiply(Matrix other) {
        if (other.rows != rows) {
            throw new IllegalArgumentException("Metrics Multiplication cannot be done");
        }
        return dotKernel(this.transpose(), other.transpose());
    }

    // this * other^T: (r x n) * (c x n)^T = r x c
    public Matrix multiplyTranspose(Matrix other) {
        if (other.cols != cols) {
            throw new IllegalArgumentException("Metrics Multiplication cannot be done");
        }
        return dotKernel(this, other);
    }

    // out[i][j] = a row i . b row j, for a (r x n) and b (c x n).
    // each a-row chunk is loaded once and multiplied against 4 b rows, so 4 independent sums keep the FMA units busy.
    // the lanes are added up by sumLanes in a fixed order, so results are the same on every run on a given machine
    // (a CPU with a different vector width rounds slightly differently)
    private static Matrix dotKernel(Matrix a, Matrix b) {
        int r = a.rows;
        int n = a.cols;
        int c = b.rows;
        float[] x = a.data;
        float[] y = b.data;
        float[] out = new float[r * c];
        int lanes = SPECIES.length();
        int vectorEnd = SPECIES.loopBound(n);
        forEachRow(r, (long) r * n * c, i -> {
            int ai = i * n;
            int oi = i * c;
            float[] lanesOut = new float[lanes];
            int j = 0;
            for (; j + 4 <= c; j += 4) {
                int b0 = j * n;
                int b1 = b0 + n;
                int b2 = b1 + n;
                int b3 = b2 + n;
                FloatVector s0 = FloatVector.zero(SPECIES);
                FloatVector s1 = s0;
                FloatVector s2 = s0;
                FloatVector s3 = s0;
                int k = 0;
                for (; k < vectorEnd; k += lanes) {
                    FloatVector av = FloatVector.fromArray(SPECIES, x, ai + k);
                    s0 = av.fma(FloatVector.fromArray(SPECIES, y, b0 + k), s0);
                    s1 = av.fma(FloatVector.fromArray(SPECIES, y, b1 + k), s1);
                    s2 = av.fma(FloatVector.fromArray(SPECIES, y, b2 + k), s2);
                    s3 = av.fma(FloatVector.fromArray(SPECIES, y, b3 + k), s3);
                }
                float t0 = sumLanes(s0, lanesOut);
                float t1 = sumLanes(s1, lanesOut);
                float t2 = sumLanes(s2, lanesOut);
                float t3 = sumLanes(s3, lanesOut);
                // n not a multiple of the lane count: the last few values one at a time
                for (; k < n; k++) {
                    float av = x[ai + k];
                    t0 += av * y[b0 + k];
                    t1 += av * y[b1 + k];
                    t2 += av * y[b2 + k];
                    t3 += av * y[b3 + k];
                }
                out[oi + j] = t0;
                out[oi + j + 1] = t1;
                out[oi + j + 2] = t2;
                out[oi + j + 3] = t3;
            }
            // c not a multiple of 4: remaining columns one at a time
            for (; j < c; j++) {
                int bj = j * n;
                FloatVector sum = FloatVector.zero(SPECIES);
                int k = 0;
                for (; k < vectorEnd; k += lanes) {
                    sum = FloatVector.fromArray(SPECIES, x, ai + k).fma(FloatVector.fromArray(SPECIES, y, bj + k), sum);
                }
                float t = sumLanes(sum, lanesOut);
                for (; k < n; k++) {
                    t += x[ai + k] * y[bj + k];
                }
                out[oi + j] = t;
            }
        });
        return new Matrix(r, c, out);
    }

    // adds the lanes left to right. reduceLanes(ADD) leaves the order open for floats, and the interpreted and
    // JIT-compiled versions really do add in different orders, which made identical runs train differently
    private static float sumLanes(FloatVector v, float[] scratch) {
        v.intoArray(scratch, 0);
        float sum = 0f;
        for (float lane : scratch) {
            sum += lane;
        }
        return sum;
    }

    public Matrix elementWiseOp(Matrix other, FloatOp func) {
        requireSameShape(other);
        float[] result = new float[data.length];
        forEachRow(rows, size(), i -> {
            for (int k = i * cols, end = k + cols; k < end; k++) {
                result[k] = func.apply(data[k], other.data[k]);
            }
        });
        return new Matrix(rows, cols, result);
    }

    // adds a 1 x cols row vector (e.g. a bias) to every row, in place
    public Matrix addRowVectorInPlace(Matrix rowVector) {
        float[] v = rowVector.data;
        if (rowVector.rows != 1 || v.length != cols) {
            throw new IllegalArgumentException(
                    "row vector is " + rowVector.rows + "x" + rowVector.cols + ", matrix has " + cols + " columns");
        }
        forEachRow(rows, size(), i -> {
            int ai = i * cols;
            for (int j = 0; j < cols; j++) {
                data[ai + j] += v[j];
            }
        });
        return this;
    }

    // 1 x cols matrix of column sums (e.g. a bias gradient); rows are added in order, in double since there can be
    // thousands of rows, and rounded to float once at the end
    public Matrix columnSums() {
        double[] sums = new double[cols];
        if (size() >= PARALLEL_THRESHOLD) {
            IntStream.range(0, cols).parallel().forEach(j -> {
                double sum = 0.0;
                for (int i = 0; i < rows; i++) {
                    sum += data[i * cols + j];
                }
                sums[j] = sum;
            });
        } else {
            for (int i = 0; i < rows; i++) {
                int ai = i * cols;
                for (int j = 0; j < cols; j++) {
                    sums[j] += data[ai + j];
                }
            }
        }
        float[] result = new float[cols];
        for (int j = 0; j < cols; j++) {
            result[j] = (float) sums[j];
        }
        return new Matrix(1, cols, result);
    }

    public Matrix softMax() {
        float[] result = new float[data.length];
        forEachRow(rows, size(), i -> {
            int ai = i * cols;
            float max = data[ai];
            for (int j = 1; j < cols; j++) {
                max = Math.max(max, data[ai + j]);
            }

            // the sum runs over up to V entries, so it adds in double
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                double e = Math.exp(data[ai + j] - max);
                result[ai + j] = (float) e;
                sum += e;
            }

            for (int j = 0; j < cols; j++) {
                result[ai + j] = (float) (result[ai + j] / sum);
            }
        });
        return new Matrix(rows, cols, result);
    }

    public Matrix relu() {
        float[] result = new float[data.length];
        forEachRow(rows, size(), i -> {
            for (int k = i * cols, end = k + cols; k < end; k++) {
                result[k] = Math.max(0.0f, data[k]);
            }
        });
        return new Matrix(rows, cols, result);
    }

    public Matrix reluAndMask() {
        float[] result = new float[data.length];
        forEachRow(rows, size(), i -> {
            for (int k = i * cols, end = k + cols; k < end; k++) {
                result[k] = data[k] > 0 ? 1f : 0f;
            }
        });
        return new Matrix(rows, cols, result);
    }

    // init helpers take the caller's Random so the same seed always gives the same weights
    public Matrix normalFill(Random rnd, double std) {
        for (int k = 0; k < data.length; k++) {
            data[k] = (float) (rnd.nextGaussian() * std);
        }
        return this;
    }

    public Matrix uniformFill(Random rnd, double bound) {
        for (int k = 0; k < data.length; k++) {
            data[k] = (float) rnd.nextDouble(-bound, bound);
        }
        return this;
    }

    public Matrix fillZeroes() {
        return fillWithValue(0.0);
    }

    public Matrix fillWithValue(double value) {
        Arrays.fill(data, (float) value);
        return this;
    }

    public Matrix transpose() {
        float[] result = new float[data.length];
        // one output row per task, so no two threads write the same row
        forEachRow(cols, size(), j -> {
            int oj = j * rows;
            for (int i = 0; i < rows; i++) {
                result[oj + i] = data[i * cols + j];
            }
        });
        return new Matrix(cols, rows, result);
    }

    public Matrix scale(double d) {
        float s = (float) d;
        float[] result = new float[data.length];
        forEachRow(rows, size(), i -> {
            for (int k = i * cols, end = k + cols; k < end; k++) {
                result[k] = data[k] * s;
            }
        });
        return new Matrix(rows, cols, result);
    }

    // Mask everything j > i, j > i is considered future case so forward pass shouldnt include that as thats cheating
    public Matrix applyCausalMask() {
        float[] result = data.clone();
        forEachRow(rows, size(), i -> {
            for (int j = i + 1; j < cols; j++) {
                result[i * cols + j] = -1e9f;
            }
        });
        return new Matrix(rows, cols, result);
    }

    public Matrix copy() {
        return new Matrix(rows, cols, data.clone());
    }

    public Matrix add(Matrix b) {
        requireSameShape(b);
        float[] result = new float[data.length];
        forEachRow(rows, size(), i -> {
            for (int k = i * cols, end = k + cols; k < end; k++) {
                result[k] = data[k] + b.data[k];
            }
        });
        return new Matrix(rows, cols, result);
    }

    private void requireSameShape(Matrix other) {
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException(
                    "shape mismatch: " + rows + "x" + cols + " vs " + other.rows + "x" + other.cols);
        }
    }

    // records compare arrays by reference, so compare the values instead
    @Override
    public boolean equals(Object o) {
        return o instanceof Matrix matrix
                && rows == matrix.rows
                && cols == matrix.cols
                && Arrays.equals(data, matrix.data);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * rows + cols) + Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            sb.append("\n");
            for (int j = 0; j < cols; j++) {
                sb.append(data[i * cols + j]).append(",");
            }
        }
        return sb.toString();
    }
}
