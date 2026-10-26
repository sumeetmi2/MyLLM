package com.myllm.decoder;

import static com.myllm.utils.TestUtils.*;

import com.myllm.attention.AttentionHead;
import com.myllm.attention.MultiHeadAttention;
import com.myllm.dao.Matrix;
import com.myllm.feedforward.FeedForward;
import com.myllm.layernorm.LayerNorm;
import java.util.Random;
import java.util.function.DoubleSupplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class DecoderBlockTest {

    // T=3, d_model=4, h=2, d_ff=8 -> all dims differ so row/col mix-ups show up
    private static final int T = 3;
    private static final int D_MODEL = 4;
    private static final int HEADS = 2;
    private static final int D_FF = 8;

    @Test
    @DisplayName("DecoderBlock forward output keeps input shape (T x d_model)")
    void forwardPassShapeTest() {
        DecoderBlock block = new DecoderBlock(D_MODEL, HEADS, D_FF, new Random(42), 0.02, 0.02);
        Matrix output = block.forwardPass(randomMatrix(new Random(42), T, D_MODEL));
        assertShape(T, D_MODEL, output);
    }

    @Test
    @DisplayName("DecoderBlock backward matches numerical gradient through every sublayer")
    void backwardPassTest() {
        Random r = new Random(42);
        DecoderBlock block = new DecoderBlock(D_MODEL, HEADS, D_FF, new Random(42), 0.02, 0.02);
        LayerNorm ln1 = block.getLn1();
        LayerNorm ln2 = block.getLn2();
        MultiHeadAttention mha = block.getMha();
        FeedForward ffn = block.getFfn();

        // default init (gamma=1, beta=0, tiny attention weights) can hide bugs, so randomise
        for (LayerNorm ln : new LayerNorm[] {ln1, ln2}) {
            fillRandom(r, ln.getGamma(), 1.0);
            fillRandom(r, ln.getBeta(), 1.0);
        }
        fillRandom(r, mha.getWO(), 0.5);
        for (AttentionHead head : mha.getAttentionHeads()) {
            fillRandom(r, head.getWQ(), 0.5);
            fillRandom(r, head.getWK(), 0.5);
            fillRandom(r, head.getWV(), 0.5);
        }
        fillRandom(r, ffn.getB_1(), 0.3);
        fillRandom(r, ffn.getB_2(), 0.3);

        Matrix x = randomMatrix(r, T, D_MODEL);
        Matrix dY = randomMatrix(r, T, D_MODEL);

        block.forwardPass(x);
        DecoderBlock.DecoderBlockBackwardResult result = block.backwardPass(dY);
        DoubleSupplier loss = weightedSumLoss(() -> block.forwardPass(x), dY);

        assertGradient("dX", x, result.dX(), loss);

        assertGradient("ln1.dGamma", ln1.getGamma(), result.ln1().dGamma(), loss);
        assertGradient("ln1.dBeta", ln1.getBeta(), result.ln1().dBeta(), loss);

        assertGradient("mha.dWo", mha.getWO(), result.mha().dWo(), loss);
        for (int h = 0; h < HEADS; h++) {
            AttentionHead head = mha.getAttentionHeads()[h];
            AttentionHead.AttentionBackwardResult headResult =
                    result.mha().perHeadGradients().get(h);
            assertGradient("mha.head" + h + ".dWq", head.getWQ(), headResult.dWq(), loss);
            assertGradient("mha.head" + h + ".dWk", head.getWK(), headResult.dWk(), loss);
            assertGradient("mha.head" + h + ".dWv", head.getWV(), headResult.dWv(), loss);
        }

        assertGradient("ln2.dGamma", ln2.getGamma(), result.ln2().dGamma(), loss);
        assertGradient("ln2.dBeta", ln2.getBeta(), result.ln2().dBeta(), loss);

        assertGradient("ffn.dW1", ffn.getW_1(), result.ffn().d_W_1(), loss);
        assertGradient("ffn.dB1", ffn.getB_1(), result.ffn().d_B_1(), loss);
        assertGradient("ffn.dW2", ffn.getW_2(), result.ffn().d_W_2(), loss);
        assertGradient("ffn.dB2", ffn.getB_2(), result.ffn().d_B_2(), loss);
    }
}
