package com.myllm.batched;

import com.myllm.dao.Matrix;
import com.myllm.decoder.DecoderBlock;
import com.myllm.feedforward.FeedForward;
import com.myllm.layernorm.LayerNorm;

/**
 * A {@code DecoderBlock} over stacked sequences.
 *
 * <p>LayerNorm and FeedForward work row by row, so the block's own layers are called directly; only attention
 * needs the batched version.
 */
public class BatchedDecoderBlock {

    private final DecoderBlock block;
    private final BatchedMultiHeadAttention mha;

    public BatchedDecoderBlock(DecoderBlock block) {
        this.block = block;
        this.mha = new BatchedMultiHeadAttention(block.getMha());
    }

    public Matrix forwardPass(Matrix input, int[] lengths) {
        Matrix normalize1 = block.getLn1().forwardPass(input);
        Matrix attention = mha.forwardPass(normalize1, lengths);
        Matrix residual1 = input.add(attention);
        Matrix normalize2 = block.getLn2().forwardPass(residual1);
        Matrix ffnOut = block.getFfn().forwardPass(normalize2);
        return residual1.add(ffnOut);
    }

    public DecoderBlock.DecoderBlockBackwardResult backwardPass(Matrix dResidual2) {
        FeedForward.FeedForwardBackwardPassResult ffnResult = block.getFfn().backwardPass(dResidual2);
        LayerNorm.LayerNormBackwardResult ln2Result = block.getLn2().backwardPass(ffnResult.d_X());
        Matrix dResidual1 = ln2Result.dX().add(dResidual2);
        var mhaResult = mha.backwardPass(dResidual1);
        LayerNorm.LayerNormBackwardResult ln1Result = block.getLn1().backwardPass(mhaResult.dX());
        Matrix dX = dResidual1.add(ln1Result.dX());
        return new DecoderBlock.DecoderBlockBackwardResult(dX, ln1Result, mhaResult, ln2Result, ffnResult);
    }
}
