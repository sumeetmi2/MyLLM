package com.myllm.transformer;

import static com.myllm.utils.TestUtils.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.myllm.attention.AttentionHead;
import com.myllm.dao.Matrix;
import com.myllm.decoder.DecoderBlock;
import com.myllm.embedding.Embedding;
import com.myllm.layernorm.LayerNorm;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.DoubleSupplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class TransformerModelTest {

    // V=7, T_max=8, T=5, d_model=4, h=2, d_ff=8, N=2 -> all dims differ, N=2 exercises block chaining
    private static final int VOCAB = 7;
    private static final int T_MAX = 8;
    private static final int D_MODEL = 4;
    private static final int HEADS = 2;
    private static final int D_FF = 8;
    private static final int BLOCKS = 2;
    // token 1 repeats -> wTE gradient must accumulate across positions
    private static final int[] TOKEN_IDS = {1, 3, 1, 0, 6};

    private TransformerModel newModel() {
        return new TransformerModel(VOCAB, T_MAX, D_MODEL, HEADS, D_FF, BLOCKS, new Random(42));
    }

    @Test
    @DisplayName("Model forward produces logits of shape (T x V)")
    void forwardPassShapeTest() {
        assertShape(TOKEN_IDS.length, VOCAB, newModel().forwardPass(TOKEN_IDS));
    }

    @Test
    @DisplayName("Softmax of each logits row sums to 1")
    void softmaxRowsSumToOneTest() {
        Matrix probabilities = newModel().forwardPass(TOKEN_IDS).softMax();
        for (double[] row : probabilities.toArray()) {
            assertEquals(1.0, Arrays.stream(row).sum(), 1e-6);
        }
    }

    @Test
    @DisplayName("Model is causal end to end: changing the last token does not change earlier logits")
    void forwardPassCausalTest() {
        TransformerModel model = newModel();
        Matrix before = model.forwardPass(TOKEN_IDS);
        int[] changedLast = TOKEN_IDS.clone();
        changedLast[changedLast.length - 1] = 2;
        Matrix after = model.forwardPass(changedLast);

        int earlier = TOKEN_IDS.length - 1;
        assertMatrixEquals(
                Arrays.copyOf(before.toArray(), earlier), Matrix.of(Arrays.copyOf(after.toArray(), earlier)), 1e-12);
    }

    @Test
    @DisplayName("Embedding rejects token ids outside the vocab")
    void invalidTokenIdTest() {
        assertThrows(IllegalArgumentException.class, () -> newModel().forwardPass(new int[] {1, VOCAB}));
        assertThrows(IllegalArgumentException.class, () -> newModel().forwardPass(new int[] {-1}));
    }

    @Test
    @DisplayName("Model backward matches numerical gradient from logits down to embeddings")
    void backwardPassTest() {
        Random r = new Random(42);
        TransformerModel model = newModel();
        Embedding embedding = model.getEmbedding();
        DecoderBlock[] blocks = model.getDecoderBlocks();
        LayerNorm finalLayerNorm = model.getFinalLayerNorm();

        // default init (gamma=1, beta=0, tiny weights) can hide bugs, so randomise
        fillRandom(r, embedding.getWTE(), 1.0);
        fillRandom(r, embedding.getWPE(), 1.0);
        fillRandom(r, finalLayerNorm.getGamma(), 1.0);
        fillRandom(r, finalLayerNorm.getBeta(), 1.0);
        for (DecoderBlock block : blocks) {
            fillRandom(r, block.getLn1().getGamma(), 1.0);
            fillRandom(r, block.getLn2().getGamma(), 1.0);
            fillRandom(r, block.getMha().getWO(), 0.5);
            for (AttentionHead head : block.getMha().getAttentionHeads()) {
                fillRandom(r, head.getWQ(), 0.5);
                fillRandom(r, head.getWK(), 0.5);
                fillRandom(r, head.getWV(), 0.5);
            }
            // keeps FFN pre-activations clear of the ReLU kink, which the 1e-2 finite-difference step would cross
            fillRandom(r, block.getFfn().getW_1(), 1.0);
        }

        Matrix dLogits = randomMatrix(r, TOKEN_IDS.length, VOCAB);
        model.forwardPass(TOKEN_IDS);
        TransformerModel.TransformerModelBackwardPassResult result = model.backwardPass(dLogits);
        DoubleSupplier loss = weightedSumLoss(() -> model.forwardPass(TOKEN_IDS), dLogits);

        assertGradient("dWUnembed", model.getWUnembed(), result.dWUnembed(), loss);
        assertGradient(
                "finalLn.dGamma",
                finalLayerNorm.getGamma(),
                result.layerNormBackwardResult().dGamma(),
                loss);
        assertGradient(
                "finalLn.dBeta",
                finalLayerNorm.getBeta(),
                result.layerNormBackwardResult().dBeta(),
                loss);

        // one weight per sublayer in every block -> proves gradient reaches each block in reverse order
        for (int b = 0; b < blocks.length; b++) {
            DecoderBlock block = blocks[b];
            DecoderBlock.DecoderBlockBackwardResult blockResult = result.blockResults()[b];
            assertGradient(
                    "block" + b + ".ln1.dGamma",
                    block.getLn1().getGamma(),
                    blockResult.ln1().dGamma(),
                    loss);
            assertGradient(
                    "block" + b + ".mha.dWo",
                    block.getMha().getWO(),
                    blockResult.mha().dWo(),
                    loss);
            assertGradient(
                    "block" + b + ".mha.head0.dWq",
                    block.getMha().getAttentionHeads()[0].getWQ(),
                    blockResult.mha().perHeadGradients().get(0).dWq(),
                    loss);
            assertGradient(
                    "block" + b + ".ln2.dBeta",
                    block.getLn2().getBeta(),
                    blockResult.ln2().dBeta(),
                    loss);
            assertGradient(
                    "block" + b + ".ffn.dW1",
                    block.getFfn().getW_1(),
                    blockResult.ffn().d_W_1(),
                    loss);
        }

        assertGradient("dWte", embedding.getWTE(), result.dEmbedding().dWte(), loss);
        assertGradient("dWpe", embedding.getWPE(), result.dEmbedding().dWpe(), loss);
    }

    @Test
    @DisplayName("parameters() and gradients() pair up: same count, same order, matching shapes, no duplicates")
    void parametersMatchGradientsTest() {
        TransformerModel model = newModel();
        model.forwardPass(TOKEN_IDS);
        List<Matrix> parameters = model.parameters();
        List<Matrix> gradients = model.backwardPass(randomMatrix(new Random(42), TOKEN_IDS.length, VOCAB))
                .gradients();

        // embedding (2) + per block [ln1 (2) + heads * qkv (3) + wO (1) + ln2 (2) + ffn (4)]
        // + final ln (2) + unembed (1)
        int expected = 2 + BLOCKS * (2 + HEADS * 3 + 1 + 2 + 4) + 3;
        assertEquals(expected, parameters.size(), "parameter count");
        assertEquals(parameters.size(), gradients.size(), "gradient count");
        for (int i = 0; i < parameters.size(); i++) {
            assertShape(parameters.get(i).rows(), parameters.get(i).cols(), gradients.get(i));
        }
        // every weight listed exactly once (a duplicate would get updated twice per step)
        Set<Matrix> unique = Collections.newSetFromMap(new IdentityHashMap<>());
        unique.addAll(parameters);
        assertEquals(parameters.size(), unique.size(), "duplicate parameter");
    }

    // sample standard deviation of all entries
    private static double std(Matrix m) {
        double sum = 0;
        double sumSq = 0;
        long n = 0;
        for (double[] row : m.toArray()) {
            for (double x : row) {
                sum += x;
                sumSq += x * x;
                n++;
            }
        }
        double mean = sum / n;
        return Math.sqrt(sumSq / n - mean * mean);
    }

    @Test
    @DisplayName("Fresh model starts near uniform predictions: loss on random targets is about ln(V)")
    void initialLossIsNearLnVTest() {
        int vocab = 512;
        int t = 64;
        TransformerModel model = new TransformerModel(vocab, t, 128, 4, 512, 4, new Random(42));
        Random r = new Random(7);
        int[] input = r.ints(t, 0, vocab).toArray();
        int[] targets = r.ints(t, 0, vocab).toArray();
        double loss = new com.myllm.crossentropy.CrossEntropyLoss().forwardPass(model.forwardPass(input), targets);
        assertEquals(Math.log(vocab), loss, 0.3);
    }

    @Test
    @DisplayName("Residual stream stays bounded through 12 GPT-2 small sized blocks at init")
    void residualStaysBoundedTest() {
        TransformerModel model = new TransformerModel(64, 64, 768, 12, 3072, 12, new Random(42));
        int[] tokens = new Random(7).ints(64, 0, 64).toArray();
        Matrix x = model.getEmbedding().forwardPass(tokens);
        for (int b = 0; b < model.getDecoderBlocks().length; b++) {
            x = model.getDecoderBlocks()[b].forwardPass(x);
            double s = std(x);
            assertTrue(s < 1.0, "residual std " + s + " after block " + b);
        }
    }

    @Test
    @DisplayName("Same seed gives identical weights, a different seed gives different weights")
    void seedDeterminesWeightsTest() {
        List<Matrix> a = new TransformerModel(VOCAB, T_MAX, D_MODEL, HEADS, D_FF, BLOCKS, new Random(1)).parameters();
        List<Matrix> b = new TransformerModel(VOCAB, T_MAX, D_MODEL, HEADS, D_FF, BLOCKS, new Random(1)).parameters();
        List<Matrix> c = new TransformerModel(VOCAB, T_MAX, D_MODEL, HEADS, D_FF, BLOCKS, new Random(2)).parameters();
        assertEquals(a, b);
        assertNotEquals(a, c);
    }

    @Test
    @DisplayName("W_o and W_2 start at 0.02/sqrt(2*blocks), the other weight matrices at 0.02")
    void residualInitIsSmallerTest() {
        int blocks = 4;
        TransformerModel model = new TransformerModel(512, 256, 128, 4, 512, blocks, new Random(42));
        double residual = 0.02 / Math.sqrt(2.0 * blocks);
        DecoderBlock block = model.getDecoderBlocks()[0];
        assertEquals(residual, std(block.getMha().getWO()), 0.1 * residual);
        assertEquals(residual, std(block.getFfn().getW_2()), 0.1 * residual);
        assertEquals(0.02, std(block.getFfn().getW_1()), 0.002);
        assertEquals(0.02, std(block.getMha().getAttentionHeads()[0].getWQ()), 0.002);
        assertEquals(0.02, std(model.getEmbedding().getWTE()), 0.002);
        assertEquals(0.02, std(model.getWUnembed()), 0.002);
    }
}
