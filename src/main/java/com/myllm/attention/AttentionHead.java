package com.myllm.attention;

import com.myllm.dao.Matrix;
import com.myllm.lora.LoraAdapter;
import java.util.Random;

/**
 * One causal self-attention head: {@code softmax(Q Kᵀ / sqrt(d_k) + mask) V}, with the backward pass written out by
 * hand.
 */
public class AttentionHead {

    private Matrix q;
    private Matrix k;
    private Matrix v;
    private Matrix wQ;
    private Matrix wK;
    private Matrix wV;
    private Matrix attnWeights;
    private int d_Model;
    private int d_k;
    private LoraAdapter loraQ;
    private LoraAdapter loraV;

    public AttentionHead(int d_Model, int d_k, Random rnd, double initStdDev) {
        this.d_k = d_k;
        this.d_Model = d_Model;
        wQ = new Matrix(d_Model, d_k).normalFill(rnd, initStdDev);
        wK = new Matrix(d_Model, d_k).normalFill(rnd, initStdDev);
        wV = new Matrix(d_Model, d_k).normalFill(rnd, initStdDev);
    }

    public void enableLora(int r, double alpha, Random rnd) {
        loraQ = new LoraAdapter(d_Model, d_k, r, alpha, rnd);
        loraV = new LoraAdapter(d_Model, d_k, r, alpha, rnd);
    }

    public LoraAdapter getLoraQ() {
        return this.loraQ;
    }

    public LoraAdapter getLoraV() {
        return this.loraV;
    }

    public Matrix forwardPass(Matrix x) {
        q = x.multiply(wQ);
        if (loraQ != null) {
            q = q.add(loraQ.forwardPass(x));
        }
        k = x.multiply(wK);
        v = x.multiply(wV);
        if (loraV != null) {
            v = v.add(loraV.forwardPass(x));
        }
        Matrix scores = q.multiplyTranspose(k).scale(1 / Math.sqrt(d_k));
        Matrix maskedScores = scores.applyCausalMask();
        attnWeights = maskedScores.softMax(); // s[i]
        return attnWeights.multiply(v);
    }

    public record AttentionBackwardResult(
            Matrix dX,
            Matrix dWq,
            Matrix dWk,
            Matrix dWv,
            LoraAdapter.LoraAdapterBackwardPassResult loraQResult,
            LoraAdapter.LoraAdapterBackwardPassResult loraVResult) {}
    // C = AB
    // dA = dC * BT
    // dB = AT * dC
    // (AB)T = BT * AT

    public AttentionBackwardResult backwardPass(Matrix x, Matrix dOutput) {
        Matrix dAttnWeights = dOutput.multiplyTranspose(v); // g[i]
        Matrix dV = attnWeights.transposeMultiply(dOutput);
        // dScores[i][j] = s[i][j] * (g[i][j] - g[i]·s[i]), the softmax backward for each row
        int T = dAttnWeights.rows();
        int cols = dAttnWeights.cols();
        float[] gData = dAttnWeights.data();
        float[] sData = attnWeights.data();
        Matrix gradientDiffFromWeightedAvgForAllGradient = dAttnWeights.copy();
        float[] diff = gradientDiffFromWeightedAvgForAllGradient.data();
        for (int i = 0; i < T; i++) {
            int start = i * cols;
            // T terms, added in double
            double dot = 0;
            for (int j = 0; j < cols; j++) {
                dot += gData[start + j] * sData[start + j];
            }
            for (int j = 0; j < cols; j++) {
                diff[start + j] = (float) (diff[start + j] - dot);
            }
        }
        Matrix dScores = attnWeights.elementWiseOp(gradientDiffFromWeightedAvgForAllGradient, (s, g) -> s * g);
        Matrix dQKT = dScores.scale(1 / Math.sqrt(d_k));
        Matrix dQ = dQKT.multiply(k);
        Matrix dK = dQKT.transposeMultiply(q);
        Matrix dWq = x.transposeMultiply(dQ);
        Matrix dWk = x.transposeMultiply(dK);
        Matrix dWv = x.transposeMultiply(dV);
        Matrix dX_FromQ = dQ.multiplyTranspose(wQ);
        Matrix dX_FromK = dK.multiplyTranspose(wK);
        Matrix dX_FromV = dV.multiplyTranspose(wV);

        LoraAdapter.LoraAdapterBackwardPassResult loraQResult = null;
        LoraAdapter.LoraAdapterBackwardPassResult loraVResult = null;
        Matrix dX = dX_FromK.add(dX_FromQ).add(dX_FromV);
        if (loraQ != null) {
            loraQResult = loraQ.backwardPass(x, dQ);
            dX = dX.add(loraQResult.dX());
        }
        if (loraV != null) {
            loraVResult = loraV.backwardPass(x, dV);
            dX = dX.add(loraVResult.dX());
        }
        return new AttentionBackwardResult(dX, dWq, dWk, dWv, loraQResult, loraVResult);
    }

    public Matrix getWQ() {
        return wQ;
    }

    public Matrix getWK() {
        return wK;
    }

    public Matrix getWV() {
        return wV;
    }

    // folds each adapter into its own weight (Q into wQ, V into wV) and drops it, so the head is a plain head again
    public void mergeLora() {
        if (loraQ != null) {
            loraQ.merge(wQ);
            loraQ = null;
        }

        if (loraV != null) {
            loraV.merge(wV);
            loraV = null;
        }
    }
}
