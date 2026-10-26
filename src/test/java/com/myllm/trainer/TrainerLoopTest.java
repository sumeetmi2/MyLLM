package com.myllm.trainer;

import static org.junit.jupiter.api.Assertions.*;

import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.optimizer.SGDOptimizer;
import com.myllm.tokenize.CharTokenizer;
import com.myllm.transformer.TransformerModel;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class TrainerLoopTest {

    private TrainerLoop newTrainer(int[] corpusIds, int vocabSize, int T, double learningRate) {
        TransformerModel model = new TransformerModel(vocabSize, T, 16, 2, 32, 2, new Random(42));
        return new TrainerLoop(
                model, new CrossEntropyLoss(), new SGDOptimizer(learningRate), corpusIds, T, new Random(42));
    }

    @Test
    @DisplayName("Window targets are the input shifted left by one")
    void windowShiftTest() {
        int[] corpus = {0, 1, 2, 3, 4, 5};
        TrainerLoop trainer = newTrainer(corpus, 6, 3, 0.1);
        TrainerLoop.Window w = trainer.window(2);
        assertArrayEquals(new int[] {2, 3, 4}, w.input());
        assertArrayEquals(new int[] {3, 4, 5}, w.targets());
    }

    @Test
    @DisplayName("Random windows always stay inside the corpus, including the last valid start")
    void randomWindowBoundsTest() {
        int[] corpus = {0, 1, 2, 3, 4, 5};
        int T = 3;
        TrainerLoop trainer = newTrainer(corpus, 6, T, 0.1);
        boolean sawLastStart = false;
        for (int i = 0; i < 500; i++) {
            TrainerLoop.Window w = trainer.randomWindow();
            assertEquals(T, w.input().length);
            assertEquals(T, w.targets().length);
            for (int t = 0; t < T; t++) {
                // corpus is 0..5 so every target is exactly input + 1
                assertEquals(w.input()[t] + 1, w.targets()[t]);
            }
            sawLastStart |= w.targets()[T - 1] == corpus[corpus.length - 1];
        }
        assertTrue(sawLastStart, "last window (ending at final token) never sampled");
    }

    @Test
    @DisplayName("Corpus shorter than T + 1 is rejected")
    void corpusTooShortTest() {
        assertThrows(IllegalArgumentException.class, () -> newTrainer(new int[] {0, 1, 2}, 3, 3, 0.1));
    }

    @Test
    @DisplayName("Training on a tiny repeated corpus drives loss down")
    void lossDecreasesTest() {
        String text = "abcabcabcabcabcabcabcabcabcabc";
        CharTokenizer tokenizer = new CharTokenizer(text);
        TrainerLoop trainer = newTrainer(tokenizer.encode(text), tokenizer.vocabSize(), 6, 0.1);

        List<Double> losses = trainer.train(300, 50);

        assertEquals(6, losses.size());
        double first = losses.get(0);
        double last = losses.get(losses.size() - 1);
        assertTrue(last < first / 2, "loss did not fall enough: first=" + first + " last=" + last);
    }
}
