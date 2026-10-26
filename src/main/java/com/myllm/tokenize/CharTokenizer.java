package com.myllm.tokenize;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * One token per distinct character, with ids in first-seen order.
 */
public class CharTokenizer implements Tokenizer {
    private final AtomicInteger counter = new AtomicInteger(0);
    private LinkedHashMap<Character, Integer> ctoi = new LinkedHashMap<>();
    private LinkedHashMap<Integer, Character> itoc = new LinkedHashMap<>();
    private static final String CORPUS_ERROR_NOT_FOUND = "No corresponding data in training corpus found";

    public CharTokenizer(String corpus) {
        for (int i = 0; i < corpus.length(); i++) {
            char c = corpus.charAt(i);
            ctoi.computeIfAbsent(c, n -> counter.getAndIncrement());
            itoc.computeIfAbsent(ctoi.get(c), a -> c);
        }
    }

    @Override
    public int[] encode(String text) {
        int[] result = new int[text.length()];
        int cnt = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (!ctoi.containsKey(c)) {
                throw new IllegalArgumentException(CORPUS_ERROR_NOT_FOUND);
            }
            result[cnt++] = ctoi.get(c);
        }
        return result;
    }

    @Override
    public String decode(int[] ids) {
        char[] result = new char[ids.length];
        int cnt = 0;
        for (int i : ids) {
            if (!itoc.containsKey(i)) {
                throw new IllegalArgumentException(CORPUS_ERROR_NOT_FOUND);
            }
            result[cnt++] = itoc.get(i);
        }
        return new String(result);
    }

    @Override
    public int vocabSize() {
        return counter.get();
    }

    @Override
    public Set<Character> unknownChars(String text) {
        Set<Character> unknown = new LinkedHashSet<>();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (!ctoi.containsKey(c)) {
                unknown.add(c);
            }
        }
        return unknown;
    }

    // unique chars in id order. Ids are assigned in first-seen order,
    // so new CharTokenizer(vocab()) rebuilds the same mapping
    public String vocab() {
        StringBuilder sb = new StringBuilder();
        for (int id = 0; id < counter.get(); id++) {
            sb.append(itoc.get(id));
        }
        return sb.toString();
    }
}
