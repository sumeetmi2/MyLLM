package com.myllm.transformer;

import com.myllm.attention.AttentionHead;
import com.myllm.dao.Matrix;
import com.myllm.decoder.DecoderBlock;
import com.myllm.embedding.Embedding;
import com.myllm.layernorm.LayerNorm;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * The decoder-only transformer: token and position embeddings, a stack of decoder blocks, a final layer norm and
 * the projection to vocabulary logits.
 *
 * <p>{@code forwardPass} caches what {@code backwardPass} needs, so a backward pass must follow the forward pass it
 * belongs to.
 */
public class TransformerModel {
    private Embedding embedding;
    private DecoderBlock[] decoderBlocks;
    private LayerNorm finalLayerNorm;
    private Matrix wUnembed;
    private Matrix xNormalised;
    private int[] cachedTokenIds;
    private final ModelConfig config;
    private boolean loraEnabled = false;
    private static final double INIT_STD = 0.02;

    // everything needed to rebuild a model with the same shapes (used when saving / loading)
    public record ModelConfig(int vocabSize, int T_max, int d_Model, int heads, int d_ff, int noOfDecoderBlocks) {}

    public TransformerModel(ModelConfig config, Random random) {
        this(
                config.vocabSize(),
                config.T_max(),
                config.d_Model(),
                config.heads(),
                config.d_ff(),
                config.noOfDecoderBlocks(),
                random);
    }

    public TransformerModel(
            int vocabSize, int T_max, int d_Model, int heads, int d_ff, int noOfDecoderBlocks, Random random) {
        config = new ModelConfig(vocabSize, T_max, d_Model, heads, d_ff, noOfDecoderBlocks);
        // weights are drawn from random in construction order, so reordering this changes the weights for a given seed.
        // W_o and W_2 add into the residual stream once per block, so they start smaller (GPT-2 scaling)
        embedding = new Embedding(vocabSize, d_Model, T_max, random, INIT_STD);
        double residualStd = INIT_STD / Math.sqrt(2.0 * config.noOfDecoderBlocks());
        decoderBlocks = new DecoderBlock[noOfDecoderBlocks];
        for (int i = 0; i < noOfDecoderBlocks; i++) {
            decoderBlocks[i] = new DecoderBlock(d_Model, heads, d_ff, random, INIT_STD, residualStd);
        }
        finalLayerNorm = new LayerNorm(d_Model);
        wUnembed = new Matrix(d_Model, vocabSize).normalFill(random, INIT_STD);
    }

    // adds LoRA adapters to W_q and W_v of every head; from then on only the adapters are trainable.
    // call before creating the optimizer, Adam sizes its state from the first parameter list it sees
    public void enableLora(int r, double alpha, Random random) {
        enableLora(r, alpha, false, random);
    }

    // allLinear also adapts W_o and the FFN's W_1 / W_2 in every block, not only W_q / W_v
    public void enableLora(int r, double alpha, boolean allLinear, Random random) {
        if (loraEnabled) {
            throw new IllegalStateException("Lora Already Enabled");
        }
        for (DecoderBlock d : decoderBlocks) {
            Arrays.stream(d.getMha().getAttentionHeads()).forEach(head -> head.enableLora(r, alpha, random));
            if (allLinear) {
                d.getMha().enableLoraO(r, alpha, random);
                d.getFfn().enableLora(r, alpha, random);
            }
        }
        loraEnabled = true;
    }

    // folds the adapters into W_q/W_v; the model is a plain model afterwards and saves normally.
    // forward output is unchanged: x·W + s·(x·A)·B = x·(W + s·A·B)
    public void mergeLora() {
        if (!loraEnabled) {
            throw new IllegalStateException("LoRA not enabled");
        }
        for (DecoderBlock d : decoderBlocks) {
            Arrays.stream(d.getMha().getAttentionHeads()).forEach(AttentionHead::mergeLora);
            d.getMha().mergeLoraO();
            d.getFfn().mergeLora();
        }
        loraEnabled = false;
    }

    public boolean isLoraEnabled() {
        return loraEnabled;
    }

    public Matrix forwardPass(int[] tokenIds) {
        cachedTokenIds = tokenIds;
        Matrix X = embedding.forwardPass(tokenIds);
        for (int i = 0; i < decoderBlocks.length; i++) {
            X = decoderBlocks[i].forwardPass(X);
        }
        xNormalised = finalLayerNorm.forwardPass(X);
        Matrix logits = xNormalised.multiply(wUnembed);
        return logits;
    }

    public record TransformerModelBackwardPassResult(
            Matrix dWUnembed,
            LayerNorm.LayerNormBackwardResult layerNormBackwardResult,
            DecoderBlock.DecoderBlockBackwardResult[] blockResults,
            Embedding.EmbeddingBackwardPassResult dEmbedding,
            boolean loraEnabled) {

        // what the optimizer updates: only the adapters when LoRA is on, otherwise every weight
        public List<Matrix> trainableGradients() {
            return loraEnabled ? loraGradients() : gradients();
        }

        // must stay in the same order as TransformerModel.loraParameters()
        public List<Matrix> loraGradients() {
            if (!loraEnabled) {
                throw new IllegalStateException("LoRA not enabled");
            }
            List<Matrix> result = new ArrayList<>();
            for (DecoderBlock.DecoderBlockBackwardResult d : blockResults) {
                for (AttentionHead.AttentionBackwardResult h : d.mha().perHeadGradients()) {
                    result.addAll(List.of(
                            h.loraQResult().dA(),
                            h.loraQResult().dB(),
                            h.loraVResult().dA(),
                            h.loraVResult().dB()));
                }
                // W_o and FFN adapters exist only with allLinear; null results mean they are off
                if (d.mha().loraOResult() != null) {
                    result.addAll(List.of(
                            d.mha().loraOResult().dA(), d.mha().loraOResult().dB()));
                }
                if (d.ffn().loraW1Result() != null) {
                    result.addAll(List.of(
                            d.ffn().loraW1Result().dA(),
                            d.ffn().loraW1Result().dB(),
                            d.ffn().loraW2Result().dA(),
                            d.ffn().loraW2Result().dB()));
                }
            }
            return result;
        }
        // must stay in the same order as TransformerModel.parameters()
        // so the optimizer pairs each weight with its gradient
        public List<Matrix> gradients() {
            List<Matrix> result = new ArrayList<>(Arrays.asList(dEmbedding.dWte(), dEmbedding.dWpe()));
            for (DecoderBlock.DecoderBlockBackwardResult b : blockResults) {
                result.addAll(Arrays.asList(b.ln1().dGamma(), b.ln1().dBeta()));
                for (AttentionHead.AttentionBackwardResult ah : b.mha().perHeadGradients()) {
                    result.addAll(Arrays.asList(ah.dWq(), ah.dWk(), ah.dWv()));
                }
                result.addAll(Arrays.asList(
                        b.mha().dWo(),
                        b.ln2().dGamma(),
                        b.ln2().dBeta(),
                        b.ffn().d_W_1(),
                        b.ffn().d_B_1(),
                        b.ffn().d_W_2(),
                        b.ffn().d_B_2()));
            }
            result.addAll(Arrays.asList(layerNormBackwardResult.dGamma(), layerNormBackwardResult.dBeta(), dWUnembed));
            return result;
        }
    }

    public TransformerModelBackwardPassResult backwardPass(Matrix dLogits) {
        Matrix dWUnembed = xNormalised.transposeMultiply(dLogits);
        Matrix dXNormalized = dLogits.multiplyTranspose(wUnembed);
        LayerNorm.LayerNormBackwardResult layerNormBackwardResult = finalLayerNorm.backwardPass(dXNormalized);
        Matrix dX = layerNormBackwardResult.dX();
        DecoderBlock.DecoderBlockBackwardResult[] results =
                new DecoderBlock.DecoderBlockBackwardResult[decoderBlocks.length];
        for (int i = decoderBlocks.length - 1; i >= 0; i--) {
            results[i] = decoderBlocks[i].backwardPass(dX);
            dX = results[i].dX();
        }
        Embedding.EmbeddingBackwardPassResult dEmbedding = embedding.backwardPass(cachedTokenIds, dX);
        return new TransformerModelBackwardPassResult(
                dWUnembed, layerNormBackwardResult, results, dEmbedding, loraEnabled);
    }

    public Embedding getEmbedding() {
        return embedding;
    }

    public DecoderBlock[] getDecoderBlocks() {
        return decoderBlocks;
    }

    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }

    public Matrix getWUnembed() {
        return wUnembed;
    }

    public ModelConfig getConfig() {
        return config;
    }

    // must stay in the same order as TransformerModelBackwardPassResult.gradients()
    public List<Matrix> parameters() {
        List<Matrix> result = new ArrayList<>(Arrays.asList(embedding.getWTE(), embedding.getWPE()));
        for (DecoderBlock d : decoderBlocks) {
            result.addAll(Arrays.asList(d.getLn1().getGamma(), d.getLn1().getBeta()));
            for (AttentionHead ah : d.getMha().getAttentionHeads()) {
                result.addAll(Arrays.asList(ah.getWQ(), ah.getWK(), ah.getWV()));
            }
            result.addAll(Arrays.asList(
                    d.getMha().getWO(),
                    d.getLn2().getGamma(),
                    d.getLn2().getBeta(),
                    d.getFfn().getW_1(),
                    d.getFfn().getB_1(),
                    d.getFfn().getW_2(),
                    d.getFfn().getB_2()));
        }
        result.addAll(Arrays.asList(finalLayerNorm.getGamma(), finalLayerNorm.getBeta(), wUnembed));
        return result;
    }

    // what the optimizer updates: only the adapters when LoRA is on, otherwise every weight
    public List<Matrix> trainableParameters() {
        return loraEnabled ? loraParameters() : parameters();
    }

    // must stay in the same order as TransformerModelBackwardPassResult.loraGradients()
    public List<Matrix> loraParameters() {
        if (!loraEnabled) {
            throw new IllegalStateException("LoRA not enabled");
        }
        List<Matrix> result = new ArrayList<>();
        for (DecoderBlock d : this.decoderBlocks) {
            for (AttentionHead h : d.getMha().getAttentionHeads()) {
                result.addAll(List.of(
                        h.getLoraQ().getA(),
                        h.getLoraQ().getB(),
                        h.getLoraV().getA(),
                        h.getLoraV().getB()));
            }
            if (d.getMha().getLoraO() != null) {
                result.addAll(List.of(
                        d.getMha().getLoraO().getA(), d.getMha().getLoraO().getB()));
            }
            if (d.getFfn().getLoraW1() != null) {
                result.addAll(List.of(
                        d.getFfn().getLoraW1().getA(),
                        d.getFfn().getLoraW1().getB(),
                        d.getFfn().getLoraW2().getA(),
                        d.getFfn().getLoraW2().getB()));
            }
        }
        return result;
    }
}
