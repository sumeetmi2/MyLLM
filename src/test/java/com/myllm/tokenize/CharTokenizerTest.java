package com.myllm.tokenize;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class CharTokenizerTest {

    CharTokenizer c = new CharTokenizer("I am Sumeet");

    @Test
    @DisplayName("char[] to int[] tokenize")
    void ctoi() {
        int[] intArr = c.encode("S m tee");
        int[] result = new int[] {4, 1, 3, 1, 7, 6, 6};
        assertArrayEquals(result, intArr);
    }

    @Test
    @DisplayName("char[] to int[] tokenize")
    void itoc() {
        int[] input = new int[] {4, 1, 3, 1, 7, 6, 6};
        String decodedString = c.decode(input);
        String result = "S m tee";
        assertEquals(result, decodedString);
    }

    @Test
    @DisplayName("not part of tokenizer should throw error")
    void tokenizerThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> c.encode("z"));
        assertThrows(IllegalArgumentException.class, () -> c.decode(new int[] {999}));
    }

    @Test
    @DisplayName("Tokenizer size")
    void tokenizerVocabSize() {
        assertEquals(8, c.vocabSize());
    }
}
