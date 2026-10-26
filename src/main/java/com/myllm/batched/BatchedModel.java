package com.myllm.batched;

import com.myllm.dao.Matrix;
import com.myllm.decoder.DecoderBlock;
import com.myllm.embedding.Embedding;
import com.myllm.layernorm.LayerNorm;
import com.myllm.transformer.TransformerModel;

/**
 * Runs a {@code TransformerModel} on several sequences at once, stacked as rows.
 *
 * <p>Wraps the model's own layers and weights (nothing is copied), so the optimizer, ModelIO, Generator and LoRA
 * merge work on the same model. {@code backwardPass} returns the model's own result type, so the
 * {@code trainableGradients()} order is unchanged. As with the model, a {@code backwardPass} must follow the
 * {@code forwardPass} it belongs to, with no other forward pass in between.
 */
public class BatchedModel {

    private final TransformerModel model;
    private final BatchedEmbedding embedding;
    private final BatchedDecoderBlock[] blocks;
    private Matrix xNormalised;
    private int[] cachedIds;
    private int[] cachedLengths;

    public BatchedModel(TransformerModel model) {
        this.model = model;
        this.embedding = new BatchedEmbedding(model.getEmbedding());
        DecoderBlock[] original = model.getDecoderBlocks();
        blocks = new BatchedDecoderBlock[original.length];
        for (int i = 0; i < original.length; i++) {
            blocks[i] = new BatchedDecoderBlock(original[i]);
        }
    }

    public Matrix forwardPass(Batch batch) {
        return forwardPass(batch.ids(), batch.lengths());
    }

    // logits for every row of every sequence: (sum of lengths) x V
    public Matrix forwardPass(int[] ids, int[] lengths) {
        int total = 0;
        for (int length : lengths) {
            if (length <= 0) {
                throw new IllegalArgumentException("sequence length must be positive, got " + length);
            }
            total += length;
        }
        if (total != ids.length) {
            throw new IllegalArgumentException("lengths sum to " + total + " but there are " + ids.length + " ids");
        }
        cachedIds = ids;
        cachedLengths = lengths;
        Matrix x = embedding.forwardPass(ids, lengths);
        for (BatchedDecoderBlock block : blocks) {
            x = block.forwardPass(x, lengths);
        }
        xNormalised = model.getFinalLayerNorm().forwardPass(x);
        return xNormalised.multiply(model.getWUnembed());
    }

    public TransformerModel.TransformerModelBackwardPassResult backwardPass(Matrix dLogits) {
        Matrix dWUnembed = xNormalised.transposeMultiply(dLogits);
        Matrix dXNormalised = dLogits.multiplyTranspose(model.getWUnembed());
        LayerNorm.LayerNormBackwardResult lnResult = model.getFinalLayerNorm().backwardPass(dXNormalised);
        Matrix dX = lnResult.dX();
        DecoderBlock.DecoderBlockBackwardResult[] results = new DecoderBlock.DecoderBlockBackwardResult[blocks.length];
        for (int i = blocks.length - 1; i >= 0; i--) {
            results[i] = blocks[i].backwardPass(dX);
            dX = results[i].dX();
        }
        Embedding.EmbeddingBackwardPassResult dEmbedding = embedding.backwardPass(cachedIds, cachedLengths, dX);
        return new TransformerModel.TransformerModelBackwardPassResult(
                dWUnembed, lnResult, results, dEmbedding, model.isLoraEnabled());
    }

    public TransformerModel getModel() {
        return model;
    }
}
