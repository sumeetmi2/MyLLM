package com.myllm.generator;

import static org.junit.jupiter.api.Assertions.*;

import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.optimizer.SGDOptimizer;
import com.myllm.tokenize.BpeTokenizer;
import com.myllm.tokenize.CharTokenizer;
import com.myllm.trainer.TrainerLoop;
import com.myllm.transformer.TransformerModel;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class GeneratorTest {

    private static final String TEXT = "abcabcabcabcabcabcabcabcabcabc";
    private static final int T_MAX = 6;

    private TransformerModel newModel(CharTokenizer tokenizer) {
        return new TransformerModel(tokenizer.vocabSize(), T_MAX, 16, 2, 32, 2, new Random(42));
    }

    @Test
    @DisplayName("argmax picks the index of the largest score")
    void argmaxTest() {
        assertEquals(2, Generator.argmax(new double[] {0.1, -3.0, 4.2, 4.1}));
        assertEquals(0, Generator.argmax(new double[] {7.0}));
    }

    @Test
    @DisplayName("Output is prompt plus n tokens, and generating past T_max does not throw")
    void lengthAndCroppingTest() {
        CharTokenizer tokenizer = new CharTokenizer(TEXT);
        Generator generator = new Generator(newModel(tokenizer), tokenizer);
        String output = generator.generate("ab", 3 * T_MAX);
        assertEquals(2 + 3 * T_MAX, output.length());
        assertTrue(output.startsWith("ab"));
    }

    @Test
    @DisplayName("Invalid prompts are rejected")
    void invalidPromptTest() {
        CharTokenizer tokenizer = new CharTokenizer(TEXT);
        Generator generator = new Generator(newModel(tokenizer), tokenizer);
        assertThrows(IllegalArgumentException.class, () -> generator.generate("", 5));
        assertThrows(IllegalArgumentException.class, () -> generator.generate("z", 5)); // not in vocab
    }

    @Test
    @DisplayName("After training on a repeated pattern, greedy generation reproduces it")
    void memorisationTest() {
        CharTokenizer tokenizer = new CharTokenizer(TEXT);
        TransformerModel model = newModel(tokenizer);
        new TrainerLoop(
                        model,
                        new CrossEntropyLoss(),
                        new SGDOptimizer(0.1),
                        tokenizer.encode(TEXT),
                        T_MAX,
                        new Random(42))
                .train(300, 100);

        assertEquals("abcabcabcabc", new Generator(model, tokenizer).generate("a", 11));
    }

    @Test
    @DisplayName("Generation stops at <|endoftext|> and leaves the marker out")
    void stopsAtEndOfTextTest() {
        // every "ab" is followed by the end-of-text marker, so the trained model should end the text right after it
        String text = ("ab" + BpeTokenizer.END_OF_TEXT).repeat(20);
        BpeTokenizer tokenizer = BpeTokenizer.train(text, 4);
        TransformerModel model = new TransformerModel(tokenizer.vocabSize(), T_MAX, 16, 2, 32, 2, new Random(42));
        new TrainerLoop(
                        model,
                        new CrossEntropyLoss(),
                        new SGDOptimizer(0.1),
                        tokenizer.encode(text),
                        T_MAX,
                        new Random(42))
                .train(300, 100);

        assertEquals("ab", new Generator(model, tokenizer).generate("ab", 10));
    }
}
