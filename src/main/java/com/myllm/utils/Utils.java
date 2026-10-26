package com.myllm.utils;

import com.myllm.dao.Matrix;

/**
 * Matrix helpers shared by the layers: joining and splitting by columns, and in-place addition.
 */
public class Utils {

    // assumes all matrices have the same shape
    public static Matrix concatMatricesByColumns(Matrix[] matrices) {
        int col = matrices[0].cols();
        int cols = col * matrices.length;
        int rows = matrices[0].rows();
        float[] result = new float[rows * cols];
        // row i of the output is row i of every matrix, side by side
        for (int i = 0; i < rows; i++) {
            for (int m = 0; m < matrices.length; m++) {
                System.arraycopy(matrices[m].data(), i * col, result, i * cols + m * col, col);
            }
        }
        return new Matrix(rows, cols, result);
    }

    public static Matrix[] reverseConcatMatricesByColumns(Matrix matrices, int numberOfMatrices) {
        int cols = matrices.cols();
        int perMatrixCol = cols / numberOfMatrices;
        int rows = matrices.rows();
        Matrix[] result = new Matrix[numberOfMatrices];
        for (int m = 0; m < result.length; m++) {
            result[m] = new Matrix(rows, perMatrixCol);
            for (int i = 0; i < rows; i++) {
                System.arraycopy(
                        matrices.data(), i * cols + m * perMatrixCol, result[m].data(), i * perMatrixCol, perMatrixCol);
            }
        }
        return result;
    }

    // Merges b into a
    public static void inPlaceAddition(Matrix a, Matrix b) {
        // without this a larger b would be added only partially, because the loop follows a's shape
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw new IllegalArgumentException(
                    "cannot add " + b.rows() + "x" + b.cols() + " into " + a.rows() + "x" + a.cols());
        }
        float[] target = a.data();
        float[] source = b.data();
        for (int k = 0; k < target.length; k++) {
            target[k] += source[k];
        }
    }
}
