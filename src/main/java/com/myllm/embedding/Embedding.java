package com.myllm.embedding;

import com.myllm.dao.Matrix;
import java.util.Random;

/**
 * Token embedding plus learned position embedding: row {@code t} is {@code wTE[id_t] + wPE[t]}.
 */
public class Embedding {

    private int v;
    private int t_Max;
    private int d_Model;
    private Matrix wTE;
    private Matrix wPE;

    public Embedding(int vocabSize, int d_Model, int t_Max, Random r, double stdDev) {
        this.v = vocabSize;
        this.t_Max = t_Max;
        this.d_Model = d_Model;
        wTE = new Matrix(v, d_Model).normalFill(r, stdDev);
        wPE = new Matrix(t_Max, d_Model).normalFill(r, stdDev);
    }

    public Matrix forwardPass(int[] inputTokenIds) {

        if (inputTokenIds.length > t_Max) {
            throw new IllegalArgumentException("t cant be greater than t_max");
        }
        int T = inputTokenIds.length;
        float[] result = new float[T * d_Model];
        float[] te = wTE.data();
        float[] pe = wPE.data();
        for (int t = 0; t < T; t++) {
            int tokenId = inputTokenIds[t];
            if (tokenId < 0 || tokenId >= v) {
                throw new IllegalArgumentException(
                        "token id " + tokenId + " at position " + t + " out of range [0," + v + ")");
            }
            int out = t * d_Model;
            int teRow = tokenId * d_Model;
            for (int j = 0; j < d_Model; j++) {
                result[out + j] = te[teRow + j] + pe[out + j];
            }
        }
        return new Matrix(T, d_Model, result);
    }

    public record EmbeddingBackwardPassResult(Matrix dWte, Matrix dWpe) {}
    // gradientMatrix is last output of the last block that generates loss gradient matrix
    public EmbeddingBackwardPassResult backwardPass(int[] tokenIds, Matrix gradientMatrix) {
        Matrix grad_wTe = new Matrix(v, d_Model).fillZeroes();
        Matrix grad_wPe = new Matrix(t_Max, d_Model).fillZeroes();
        float[] dX = gradientMatrix.data();
        float[] dTe = grad_wTe.data();
        float[] dPe = grad_wPe.data();
        for (int t = 0; t < tokenIds.length; t++) {
            int row = t * d_Model;
            int teRow = tokenIds[t] * d_Model;
            for (int j = 0; j < d_Model; j++) {
                dTe[teRow + j] += dX[row + j];
                dPe[row + j] += dX[row + j];
            }
        }

        return new EmbeddingBackwardPassResult(grad_wTe, grad_wPe);
    }

    public Matrix getWTE() {
        return wTE;
    }

    public Matrix getWPE() {
        return wPE;
    }

    public int getTMax() {
        return t_Max;
    }
}
