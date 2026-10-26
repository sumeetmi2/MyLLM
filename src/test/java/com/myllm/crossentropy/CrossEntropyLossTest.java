package com.myllm.crossentropy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.myllm.dao.Matrix;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class CrossEntropyLossTest {

    @Test
    @DisplayName("Loss stays finite when the target's probability underflows float")
    void extremeLogitsTest() {
        // p(target) = e^-200, far below float's smallest value, so -log(p) on the stored probability would be infinite
        Matrix logits = Matrix.of(new double[][] {{100, -100, -100}});
        CrossEntropyLoss loss = new CrossEntropyLoss();
        double value = loss.forwardPass(logits, new int[] {1});
        assertEquals(200.0, value, 1e-3);

        Matrix dLogits = loss.backwardPass();
        for (float g : dLogits.data()) {
            assertTrue(Float.isFinite(g), "gradient " + g);
        }
        // (p - onehot): the target column gets about -1, the confident column about +1
        assertEquals(-1.0, dLogits.get(0, 1), 1e-6);
        assertEquals(1.0, dLogits.get(0, 0), 1e-6);
    }
}
