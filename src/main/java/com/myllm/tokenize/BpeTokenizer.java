package com.myllm.tokenize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * Byte pair encoding over characters.
 *
 * <p>Training starts from single chars and keeps merging the most frequent adjacent pair into a new token. Token
 * ids: {@code [0, B)} base chars in first-seen order, then {@code <|endoftext|>} (if the corpus has it), then one id
 * per merge.
 */
public class BpeTokenizer implements Tokenizer {

    public static final String END_OF_TEXT = "<|endoftext|>";

    // two adjacent token ids
    public record Pair(int first, int second) {}

    // Text is split into chunks before merging, and merges never cross chunks (simplified GPT-2 rule):
    // a word keeps its leading space (" cat"), numbers and punctuation are separate, whitespace runs stay together.
    private static final Pattern CHUNK = Pattern.compile(" ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+(?!\\S)|\\s+");

    private final String baseChars;
    private final boolean hasEndOfText;
    private final List<Pair> merges;

    private final Map<Character, Integer> charToId = new HashMap<>();
    private final Map<Pair, Integer> mergeRank = new HashMap<>();
    private final String[] tokenText;
    private final int endOfTextId;
    private final int firstMergeId;
    // chunks repeat constantly in real text, so each one is encoded once
    private final Map<String, int[]> chunkCache = new HashMap<>();

    // rebuilds a trained tokenizer from its parts (used by train() and when loading a saved model)
    public BpeTokenizer(String baseChars, boolean hasEndOfText, List<Pair> merges) {
        this.baseChars = baseChars;
        this.hasEndOfText = hasEndOfText;
        this.merges = List.copyOf(merges);
        this.endOfTextId = hasEndOfText ? baseChars.length() : -1;
        this.firstMergeId = baseChars.length() + (hasEndOfText ? 1 : 0);
        this.tokenText = new String[firstMergeId + merges.size()];

        for (int id = 0; id < baseChars.length(); id++) {
            char c = baseChars.charAt(id);
            if (charToId.put(c, id) != null) {
                throw new IllegalArgumentException("duplicate base char '" + c + "'");
            }
            tokenText[id] = String.valueOf(c);
        }
        if (hasEndOfText) {
            tokenText[endOfTextId] = END_OF_TEXT;
        }
        for (int rank = 0; rank < merges.size(); rank++) {
            Pair pair = merges.get(rank);
            int id = firstMergeId + rank;
            checkMergePart(pair.first(), id);
            checkMergePart(pair.second(), id);
            if (mergeRank.put(pair, rank) != null) {
                throw new IllegalArgumentException("merge " + rank + " repeats an earlier pair");
            }
            tokenText[id] = tokenText[pair.first()] + tokenText[pair.second()];
        }
    }

    // a merge may only combine tokens that already exist, and never the special token
    private void checkMergePart(int part, int mergeId) {
        if (part < 0 || part >= mergeId || part == endOfTextId) {
            throw new IllegalArgumentException("merge for id " + mergeId + " uses invalid id " + part);
        }
    }

    // training

    public static BpeTokenizer train(String text, int vocabSize) {
        return train(text, vocabSize, text.length());
    }

    // Base chars come from the whole text, so every char stays encodable.
    // Merges are learned from about the first sampleChars chars, which is much faster on big corpora.
    public static BpeTokenizer train(String text, int vocabSize, int sampleChars) {
        BpeTokenizer tokenizer = new BpeTokenizer(distinctChars(text), text.contains(END_OF_TEXT), List.of());
        if (vocabSize < tokenizer.vocabSize()) {
            throw new IllegalArgumentException(
                    "vocabSize " + vocabSize + " is below the " + tokenizer.vocabSize() + " base tokens");
        }
        List<Word> words = tokenizer.distinctWords(sample(text, sampleChars));

        List<Pair> merges = new ArrayList<>();
        while (tokenizer.vocabSize() + merges.size() < vocabSize) {
            Map<Pair, Long> pairCounts = countPairs(words);
            Pair best = mostFrequent(pairCounts);
            // a pair seen only once is not worth a token; the vocab stays smaller than asked
            if (best == null || pairCounts.get(best) < 2) {
                break;
            }
            int newId = tokenizer.vocabSize() + merges.size();
            words.replaceAll(word -> word.merge(best, newId));
            merges.add(best);
        }
        return new BpeTokenizer(tokenizer.baseChars, tokenizer.hasEndOfText, merges);
    }

    // one distinct chunk of the training text, as token ids, plus how often it occurs
    private record Word(int[] ids, int count) {
        Word merge(Pair pair, int newId) {
            return new Word(mergePair(ids, pair, newId), count);
        }
    }

    // merging works on distinct chunks with counts instead of the raw text: far fewer symbols to scan
    private List<Word> distinctWords(String text) {
        Map<String, Integer> chunkCounts = new LinkedHashMap<>();
        for (String part : splitOnEndOfText(text)) {
            for (String chunk : chunks(part)) {
                chunkCounts.merge(chunk, 1, Integer::sum);
            }
        }
        List<Word> words = new ArrayList<>();
        chunkCounts.forEach((chunk, count) -> words.add(new Word(charIds(chunk), count)));
        return words;
    }

    private static Map<Pair, Long> countPairs(List<Word> words) {
        Map<Pair, Long> counts = new HashMap<>();
        for (Word word : words) {
            for (int i = 0; i + 1 < word.ids().length; i++) {
                counts.merge(new Pair(word.ids()[i], word.ids()[i + 1]), (long) word.count(), Long::sum);
            }
        }
        return counts;
    }

    // highest count wins; ties go to the smallest ids so training is deterministic
    private static Pair mostFrequent(Map<Pair, Long> pairCounts) {
        Comparator<Map.Entry<Pair, Long>> order = Map.Entry.<Pair, Long>comparingByValue()
                .thenComparingInt(e -> -e.getKey().first())
                .thenComparingInt(e -> -e.getKey().second());
        return pairCounts.entrySet().stream().max(order).map(Map.Entry::getKey).orElse(null);
    }

    // encode / decode

    @Override
    public int[] encode(String text) {
        List<String> parts = splitOnEndOfText(text);
        if (!hasEndOfText && parts.size() > 1) {
            throw new IllegalArgumentException(END_OF_TEXT + " is not in this tokenizer's vocab");
        }
        IntStream.Builder ids = IntStream.builder();
        for (int p = 0; p < parts.size(); p++) {
            if (p > 0) {
                ids.add(endOfTextId);
            }
            for (String chunk : chunks(parts.get(p))) {
                for (int id : encodeChunk(chunk)) {
                    ids.add(id);
                }
            }
        }
        return ids.build().toArray();
    }

    // start from single chars, then apply merges in the order they were learned (lowest rank first),
    // so encoding rebuilds exactly the tokens that training built
    private int[] encodeChunk(String chunk) {
        return chunkCache.computeIfAbsent(chunk, c -> {
            int[] ids = charIds(c);
            while (true) {
                Pair next = lowestRankPair(ids);
                if (next == null) {
                    return ids;
                }
                ids = mergePair(ids, next, firstMergeId + mergeRank.get(next));
            }
        });
    }

    private Pair lowestRankPair(int[] ids) {
        Pair best = null;
        for (int i = 0; i + 1 < ids.length; i++) {
            Pair pair = new Pair(ids[i], ids[i + 1]);
            Integer rank = mergeRank.get(pair);
            if (rank != null && (best == null || rank < mergeRank.get(best))) {
                best = pair;
            }
        }
        return best;
    }

    @Override
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= tokenText.length) {
                throw new IllegalArgumentException("token id " + id + " out of range [0," + tokenText.length + ")");
            }
            sb.append(tokenText[id]);
        }
        return sb.toString();
    }

    @Override
    public int vocabSize() {
        return tokenText.length;
    }

    @Override
    public int endOfTextId() {
        return endOfTextId;
    }

    @Override
    public Set<Character> unknownChars(String text) {
        String plain = hasEndOfText ? text.replace(END_OF_TEXT, "") : text;
        Set<Character> unknown = new LinkedHashSet<>();
        for (int i = 0; i < plain.length(); i++) {
            char c = plain.charAt(i);
            if (!charToId.containsKey(c)) {
                unknown.add(c);
            }
        }
        return unknown;
    }

    // accessors (for ModelIO and tests)

    public String tokenText(int id) {
        return tokenText[id];
    }

    public String baseChars() {
        return baseChars;
    }

    public boolean hasEndOfText() {
        return hasEndOfText;
    }

    public List<Pair> merges() {
        return merges;
    }

    // helpers

    private int[] charIds(String chunk) {
        int[] ids = new int[chunk.length()];
        for (int i = 0; i < ids.length; i++) {
            Integer id = charToId.get(chunk.charAt(i));
            if (id == null) {
                throw new IllegalArgumentException("character '" + chunk.charAt(i) + "' is not in the tokenizer vocab");
            }
            ids[i] = id;
        }
        return ids;
    }

    static List<String> chunks(String text) {
        List<String> result = new ArrayList<>();
        Matcher m = CHUNK.matcher(text);
        while (m.find()) {
            result.add(m.group());
        }
        return result;
    }

    // replaces every (first, second) with merged, left to right, without overlaps
    private static int[] mergePair(int[] ids, Pair pair, int merged) {
        int[] out = new int[ids.length];
        int n = 0;
        int i = 0;
        while (i < ids.length) {
            if (i + 1 < ids.length && ids[i] == pair.first() && ids[i + 1] == pair.second()) {
                out[n++] = merged;
                i += 2;
            } else {
                out[n++] = ids[i];
                i++;
            }
        }
        return Arrays.copyOf(out, n);
    }

    // the text between <|endoftext|> markers; a single part when there is none
    private static List<String> splitOnEndOfText(String text) {
        return Arrays.asList(text.split(Pattern.quote(END_OF_TEXT), -1));
    }

    // every char except those that only appear inside <|endoftext|>, in first-seen order
    private static String distinctChars(String text) {
        StringBuilder chars = new StringBuilder();
        text.replace(END_OF_TEXT, "").chars().distinct().forEach(c -> chars.append((char) c));
        return chars.toString();
    }

    // cut at a newline so the sample never ends inside a word or a special token
    private static String sample(String text, int sampleChars) {
        if (sampleChars >= text.length()) {
            return text;
        }
        int cut = text.lastIndexOf('\n', sampleChars);
        return text.substring(0, cut > 0 ? cut : sampleChars);
    }
}
