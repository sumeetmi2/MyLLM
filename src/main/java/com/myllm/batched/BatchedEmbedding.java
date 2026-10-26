package com.myllm.batched;

import com.myllm.dao.Matrix;
import com.myllm.embedding.Embedding;

/**
 * Embedding for stacked sequences: positions restart at 0 for each sequence. Reads the wrapped layer's weights.
 */
public class BatchedEmbedding {

    private final Embedding embedding;

    public BatchedEmbedding(Embedding embedding) {
        this.embedding = embedding;
    }

    public Matrix forwardPass(int[] ids, int[] lengths) {
        Matrix wTE = embedding.getWTE();
        int v = wTE.rows();
        int d = wTE.cols();
        float[] te = wTE.data();
        float[] pe = embedding.getWPE().data();
        float[] result = new float[ids.length * d];
        int row = 0;
        for (int length : lengths) {
            if (length > embedding.getTMax()) {
                throw new IllegalArgumentException(
                        "sequence of " + length + " tokens is longer than t_max " + embedding.getTMax());
            }
            for (int t = 0; t < length; t++, row++) {
                int tokenId = ids[row];
                if (tokenId < 0 || tokenId >= v) {
                    throw new IllegalArgumentException(
                            "token id " + tokenId + " at row " + row + " out of range [0," + v + ")");
                }
                int out = row * d;
                int teRow = tokenId * d;
                int peRow = t * d;
                for (int j = 0; j < d; j++) {
                    result[out + j] = te[teRow + j] + pe[peRow + j];
                }
            }
        }
        return new Matrix(ids.length, d, result);
    }

    // every sequence adds into the same position rows of dWpe
    public Embedding.EmbeddingBackwardPassResult backwardPass(int[] ids, int[] lengths, Matrix gradientMatrix) {
        int d = embedding.getWTE().cols();
        Matrix grad_wTe = new Matrix(embedding.getWTE().rows(), d);
        Matrix grad_wPe = new Matrix(embedding.getTMax(), d);
        float[] dX = gradientMatrix.data();
        float[] dTe = grad_wTe.data();
        float[] dPe = grad_wPe.data();
        int row = 0;
        for (int length : lengths) {
            for (int t = 0; t < length; t++, row++) {
                int src = row * d;
                int teRow = ids[row] * d;
                int peRow = t * d;
                for (int j = 0; j < d; j++) {
                    dTe[teRow + j] += dX[src + j];
                    dPe[peRow + j] += dX[src + j];
                }
            }
        }
        return new Embedding.EmbeddingBackwardPassResult(grad_wTe, grad_wPe);
    }
}
