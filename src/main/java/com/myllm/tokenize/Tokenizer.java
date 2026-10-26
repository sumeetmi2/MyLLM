package com.myllm.tokenize;

import java.util.Set;

/**
 * Maps text to token ids and back. The model only sees ids, so any tokenizer works with it.
 */
public interface Tokenizer {

    int[] encode(String text);

    String decode(int[] ids);

    int vocabSize();

    // characters in text that this tokenizer can't encode (encode would throw on them)
    Set<Character> unknownChars(String text);

    // id of the single token that marks the end of a text, or -1 when this tokenizer has none
    default int endOfTextId() {
        return -1;
    }
}
