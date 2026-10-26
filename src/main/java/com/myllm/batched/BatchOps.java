package com.myllm.batched;

import com.myllm.dao.Matrix;

/**
 * Row slicing for stacked sequences. Rows are contiguous in {@code Matrix.data}, so each slice is one array copy.
 */
final class BatchOps {

    private BatchOps() {}

    // copy of rows [from, from + count)
    static Matrix rows(Matrix m, int from, int count) {
        Matrix out = new Matrix(count, m.cols());
        System.arraycopy(m.data(), m.offset(from), out.data(), 0, count * m.cols());
        return out;
    }

    // writes source into target starting at row `from`
    static void writeRows(Matrix target, int from, Matrix source) {
        System.arraycopy(source.data(), 0, target.data(), target.offset(from), source.data().length);
    }

    static int[] starts(int[] lengths) {
        int[] starts = new int[lengths.length];
        for (int s = 1; s < lengths.length; s++) {
            starts[s] = starts[s - 1] + lengths[s - 1];
        }
        return starts;
    }
}
