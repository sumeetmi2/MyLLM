package com.myllm.generator;

import static org.junit.jupiter.api.Assertions.*;

import com.myllm.tokenize.CharTokenizer;
import com.myllm.transformer.TransformerModel;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class InteractiveSessionTest {

    private static final String TEXT = "abcabcabc";

    private String runSession(String input, int length) throws IOException {
        CharTokenizer tokenizer = new CharTokenizer(TEXT);
        TransformerModel model = new TransformerModel(tokenizer.vocabSize(), 6, 8, 2, 16, 1, new Random(42));
        InteractiveSession session = new InteractiveSession(new Generator(model, tokenizer), tokenizer, length);
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        session.run(new BufferedReader(new StringReader(input)), new PrintStream(buffer, true, StandardCharsets.UTF_8));
        return buffer.toString(StandardCharsets.UTF_8);
    }

    @Test
    @DisplayName("Each prompt line gets a continuation of prompt + length chars")
    void generatesForEachLineTest() throws IOException {
        String[] lines = runSession("ab\nc\n", 4).split("\n");
        // banner, "> ab"+4 chars, "> c"+4 chars, then "> " before end of input
        assertTrue(lines[1].startsWith("> ab"));
        assertEquals("> ".length() + 2 + 4, lines[1].length());
        assertTrue(lines[2].startsWith("> c"));
        assertEquals("> ".length() + 1 + 4, lines[2].length());
    }

    @Test
    @DisplayName("Characters outside the vocab are reported instead of crashing")
    void unknownCharactersTest() throws IOException {
        String output = runSession("abz!\n", 4);
        assertTrue(output.contains("not in model vocab: [z, !]"));
    }

    @Test
    @DisplayName("/quit ends the session before later lines")
    void quitTest() throws IOException {
        String output = runSession("/quit\nab\n", 4);
        assertFalse(output.contains("> ab"));
    }
}
