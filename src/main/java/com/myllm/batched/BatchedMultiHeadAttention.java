package com.myllm.batched;

import com.myllm.attention.AttentionHead;
import com.myllm.attention.MultiHeadAttention;
import com.myllm.dao.Matrix;
import com.myllm.lora.LoraAdapter;
import com.myllm.utils.Utils;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Multi-head attention over stacked sequences: batched heads, then the same concat and {@code W_o} as the
 * single-sequence version.
 */
public class BatchedMultiHeadAttention {

    private final MultiHeadAttention mha;
    private final BatchedAttentionHead[] heads;
    private Matrix combined;
    private Matrix cachedX;

    public BatchedMultiHeadAttention(MultiHeadAttention mha) {
        this.mha = mha;
        AttentionHead[] original = mha.getAttentionHeads();
        heads = new BatchedAttentionHead[original.length];
        for (int i = 0; i < original.length; i++) {
            heads[i] = new BatchedAttentionHead(original[i]);
        }
    }

    public Matrix forwardPass(Matrix x, int[] lengths) {
        this.cachedX = x;
        Matrix[] outputs = new Matrix[heads.length];
        IntStream.range(0, heads.length).parallel().forEach(i -> outputs[i] = heads[i].forwardPass(x, lengths));
        combined = Utils.concatMatricesByColumns(outputs);
        Matrix output = combined.multiply(mha.getWO());
        LoraAdapter loraO = mha.getLoraO();
        if (loraO != null) {
            output = output.add(loraO.forwardPass(combined));
        }
        return output;
    }

    public MultiHeadAttention.MHABackwardResult backwardPass(Matrix dOutput) {
        Matrix dCombined = dOutput.multiplyTranspose(mha.getWO());
        LoraAdapter loraO = mha.getLoraO();
        LoraAdapter.LoraAdapterBackwardPassResult loraOResult = null;
        if (loraO != null) {
            loraOResult = loraO.backwardPass(combined, dOutput);
            dCombined = dCombined.add(loraOResult.dX());
        }
        Matrix dWO = combined.transposeMultiply(dOutput);
        Matrix[] sliced = Utils.reverseConcatMatricesByColumns(dCombined, heads.length);
        AttentionHead.AttentionBackwardResult[] perHead = new AttentionHead.AttentionBackwardResult[heads.length];
        IntStream.range(0, heads.length)
                .parallel()
                .forEach(i -> perHead[i] = heads[i].backwardPass(cachedX, sliced[i]));
        // head order matters: the optimizer pairs gradients with parameters by position
        List<AttentionHead.AttentionBackwardResult> perHeadList = List.of(perHead);
        Matrix dX = perHeadList.stream()
                .map(AttentionHead.AttentionBackwardResult::dX)
                .reduce(Matrix::add)
                .get();
        return new MultiHeadAttention.MHABackwardResult(dX, dWO, perHeadList, loraOResult);
    }
}
