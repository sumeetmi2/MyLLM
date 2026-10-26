package com.myllm.attention;

import com.myllm.dao.Matrix;
import com.myllm.lora.LoraAdapter;
import com.myllm.utils.Utils;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Multi-head causal self-attention: runs the heads side by side, concatenates their outputs and projects them
 * with {@code W_o}.
 */
public class MultiHeadAttention {

    private Matrix wO;
    private AttentionHead[] attentionHeads;
    private Matrix combinedMatricesByColumns = null;
    private int d_Model;
    private int noOfAttentionHeads;
    private Matrix cachedX;
    // null = LoRA off; reads the concatenated head outputs, like W_o
    private LoraAdapter loraO;

    public MultiHeadAttention(
            int d_Model, int noOfAttentionHeads, Random rnd, double initStdDev, double residualStdDev) {
        if (noOfAttentionHeads <= 0 || d_Model % noOfAttentionHeads != 0) {
            throw new IllegalArgumentException(
                    "d_Model (" + d_Model + ") must be divisible by number of heads (" + noOfAttentionHeads + ")");
        }
        this.d_Model = d_Model;
        this.noOfAttentionHeads = noOfAttentionHeads;
        int d_K = d_Model / noOfAttentionHeads;
        attentionHeads = new AttentionHead[noOfAttentionHeads];
        for (int i = 0; i < attentionHeads.length; i++) {
            attentionHeads[i] = new AttentionHead(d_Model, d_K, rnd, initStdDev);
        }
        wO = new Matrix(d_Model, d_Model).normalFill(rnd, residualStdDev);
    }

    public void enableLoraO(int r, double alpha, Random rnd) {
        loraO = new LoraAdapter(d_Model, d_Model, r, alpha, rnd);
    }

    // folds the W_o adapter (if any) into W_o and drops it
    public void mergeLoraO() {
        if (loraO != null) {
            loraO.merge(wO);
            loraO = null;
        }
    }

    public LoraAdapter getLoraO() {
        return loraO;
    }

    public Matrix forwardPass(Matrix x) {
        this.cachedX = x;
        // heads keep their own caches, so they can run at the same time
        Matrix[] attentionHeadOutputs = new Matrix[attentionHeads.length];
        IntStream.range(0, attentionHeads.length)
                .parallel()
                .forEach(i -> attentionHeadOutputs[i] = attentionHeads[i].forwardPass(x));
        combinedMatricesByColumns = Utils.concatMatricesByColumns(attentionHeadOutputs); // T * (h*d_k) = T * d_model
        Matrix output = combinedMatricesByColumns.multiply(wO);
        if (loraO != null) {
            output = output.add(loraO.forwardPass(combinedMatricesByColumns));
        }
        return output;
    }

    public record MHABackwardResult(
            Matrix dX,
            Matrix dWo,
            List<AttentionHead.AttentionBackwardResult> perHeadGradients,
            LoraAdapter.LoraAdapterBackwardPassResult loraOResult) {}

    public MHABackwardResult backwardPass(Matrix dOutput) {
        Matrix dCombinedByColumns = dOutput.multiplyTranspose(wO);
        // the W_o adapter also read the concatenated heads, so its share goes in before splitting per head
        LoraAdapter.LoraAdapterBackwardPassResult loraOResult = null;
        if (loraO != null) {
            loraOResult = loraO.backwardPass(combinedMatricesByColumns, dOutput);
            dCombinedByColumns = dCombinedByColumns.add(loraOResult.dX());
        }
        Matrix dWO = combinedMatricesByColumns.transposeMultiply(dOutput);
        Matrix[] slicedCombinedMatricesByColumns =
                Utils.reverseConcatMatricesByColumns(dCombinedByColumns, noOfAttentionHeads);
        AttentionHead.AttentionBackwardResult[] perHead =
                new AttentionHead.AttentionBackwardResult[attentionHeads.length];
        IntStream.range(0, attentionHeads.length)
                .parallel()
                .forEach(i ->
                        perHead[i] = attentionHeads[i].backwardPass(this.cachedX, slicedCombinedMatricesByColumns[i]));
        // kept in head order: the optimizer pairs gradients with parameters by position
        List<AttentionHead.AttentionBackwardResult> perHeadGradientLoss_Q_K_V_Result = List.of(perHead);
        Matrix dX_total = perHeadGradientLoss_Q_K_V_Result.stream()
                .map(mm -> mm.dX())
                .reduce(Matrix::add)
                .get();
        return new MHABackwardResult(dX_total, dWO, perHeadGradientLoss_Q_K_V_Result, loraOResult);
    }

    public Matrix getWO() {
        return wO;
    }

    public AttentionHead[] getAttentionHeads() {
        return attentionHeads;
    }
}
