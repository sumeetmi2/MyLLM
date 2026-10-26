package com.myllm.decoder;

import com.myllm.attention.MultiHeadAttention;
import com.myllm.dao.Matrix;
import com.myllm.feedforward.FeedForward;
import com.myllm.layernorm.LayerNorm;
import java.util.Random;

/**
 * One pre-norm decoder block: {@code x + attention(ln1(x))}, then {@code h + ffn(ln2(h))}.
 */
public class DecoderBlock {
    private LayerNorm ln1;
    private LayerNorm ln2;
    private MultiHeadAttention mha;
    private FeedForward ffn;

    public DecoderBlock(int d_Model, int heads, int d_ff, Random rnd, double initStdDev, double residualStdDev) {
        this.ln1 = new LayerNorm(d_Model);
        this.ln2 = new LayerNorm(d_Model);
        this.mha = new MultiHeadAttention(d_Model, heads, rnd, initStdDev, residualStdDev);
        this.ffn = new FeedForward(d_Model, d_ff, rnd, initStdDev, residualStdDev);
    }

    public Matrix forwardPass(Matrix input) {
        Matrix normalize1 = ln1.forwardPass(input);
        Matrix attention = mha.forwardPass(normalize1);
        Matrix residual1 = input.add(attention);
        Matrix normalize2 = ln2.forwardPass(residual1);
        Matrix ffnOut = ffn.forwardPass(normalize2);
        Matrix residual2 = residual1.add(ffnOut);
        return residual2;
    }

    public record DecoderBlockBackwardResult(
            Matrix dX,
            LayerNorm.LayerNormBackwardResult ln1,
            MultiHeadAttention.MHABackwardResult mha,
            LayerNorm.LayerNormBackwardResult ln2,
            FeedForward.FeedForwardBackwardPassResult ffn) {}

    public DecoderBlockBackwardResult backwardPass(Matrix dResidual2) {
        // residual2 = residual1 + ffnOut, so dResidual2 flows unchanged into the FFN branch (and the skip path below)
        FeedForward.FeedForwardBackwardPassResult feedForwardBackwardPassResult = ffn.backwardPass(dResidual2);
        LayerNorm.LayerNormBackwardResult layerNormBackwardResult2 =
                ln2.backwardPass(feedForwardBackwardPassResult.d_X());
        Matrix dResidual1 = layerNormBackwardResult2.dX().add(dResidual2);
        MultiHeadAttention.MHABackwardResult dAttentionResult = mha.backwardPass(dResidual1);
        LayerNorm.LayerNormBackwardResult layerNormBackwardResult1 = ln1.backwardPass(dAttentionResult.dX());
        Matrix dX = dResidual1.add(layerNormBackwardResult1.dX());
        return new DecoderBlockBackwardResult(
                dX,
                layerNormBackwardResult1,
                dAttentionResult,
                layerNormBackwardResult2,
                feedForwardBackwardPassResult);
    }

    public LayerNorm getLn1() {
        return ln1;
    }

    public MultiHeadAttention getMha() {
        return mha;
    }

    public LayerNorm getLn2() {
        return ln2;
    }

    public FeedForward getFfn() {
        return ffn;
    }
}
