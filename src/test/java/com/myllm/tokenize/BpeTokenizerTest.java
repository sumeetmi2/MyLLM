package com.myllm.tokenize;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class BpeTokenizerTest {

    private static final String STORY = """
            Once upon a time, there was a little girl named Lily. She liked to play in the park.
            One day, Lily saw a big dog. The dog was very happy. "Can I play with you?" Lily asked.
            The dog wagged its tail. They played all day and were very happy.
            """.repeat(5);

    @Test
    @DisplayName("Chunks keep the leading space on words and split punctuation off")
    void chunksTest() {
        assertEquals(
                List.of("The", " cat", " said", ",", " \"", "hi", "!\"", "\n", "Then"),
                BpeTokenizer.chunks("The cat said, \"hi!\"\nThen"));
        assertEquals("a  b\n\n c", String.join("", BpeTokenizer.chunks("a  b\n\n c")));
    }

    @Test
    @DisplayName("Toy corpus learns the most frequent pairs first, ties broken by smallest ids")
    void toyMergesTest() {
        // base ids in first-seen order: l=0 o=1 w=2 ' '=3 e=4 r=5 s=6 t=7
        BpeTokenizer bpe = BpeTokenizer.train("low low lower lowest", 11);
        assertEquals(
                List.of(
                        new BpeTokenizer.Pair(0, 1), // (l,o) count 4, ties with (o,w) but has smaller ids
                        new BpeTokenizer.Pair(8, 2), // (lo,w) count 4
                        new BpeTokenizer.Pair(3, 9)), // (' ',low) count 3
                bpe.merges());
        assertEquals(" low", bpe.tokenText(10));
    }

    @Test
    @DisplayName("decode(encode(s)) gives back s, for training text and unseen text")
    void roundTripTest() {
        BpeTokenizer bpe = BpeTokenizer.train(STORY, 120);
        assertEquals(STORY, bpe.decode(bpe.encode(STORY)));
        String unseen = "The happy girl named Lily asked the dog to play.";
        assertEquals(unseen, bpe.decode(bpe.encode(unseen)));
    }

    @Test
    @DisplayName("English text needs far fewer BPE tokens than characters")
    void compressionTest() {
        BpeTokenizer bpe = BpeTokenizer.train(STORY, 200);
        int tokens = bpe.encode(STORY).length;
        assertTrue(tokens * 3 <= STORY.length(), STORY.length() + " chars became " + tokens + " tokens");
    }

    @Test
    @DisplayName("Vocab size is base chars + merges, and training stops early when no pair repeats")
    void vocabSizeTest() {
        BpeTokenizer bpe = BpeTokenizer.train(STORY, 100);
        assertEquals(100, bpe.vocabSize());
        assertEquals(bpe.baseChars().length() + bpe.merges().size(), bpe.vocabSize());

        BpeTokenizer tiny = BpeTokenizer.train("abcd", 50);
        assertEquals(4, tiny.vocabSize());
        assertThrows(IllegalArgumentException.class, () -> BpeTokenizer.train("abcd", 3));
    }

    @Test
    @DisplayName("Encoding applies merges in learned order, not left to right")
    void rankOrderTest() {
        // a=0 b=1 c=2; merge 0 = (b,c) -> 3, merge 1 = (a,b) -> 4
        BpeTokenizer bpe =
                new BpeTokenizer("abc", false, List.of(new BpeTokenizer.Pair(1, 2), new BpeTokenizer.Pair(0, 1)));
        assertArrayEquals(new int[] {0, 3}, bpe.encode("abc"));
        assertArrayEquals(new int[] {4}, bpe.encode("ab"));
    }

    @Test
    @DisplayName("No token spans a word boundary: whitespace only at the start, or the whole token")
    void noCrossChunkTokensTest() {
        BpeTokenizer bpe = BpeTokenizer.train(STORY, 200);
        for (int id = 0; id < bpe.vocabSize(); id++) {
            String text = bpe.tokenText(id);
            boolean allWhitespace = text.isBlank();
            assertTrue(allWhitespace || !text.substring(1).matches(".*\\s.*"), "token " + id + " = [" + text + "]");
        }
    }

    @Test
    @DisplayName("Unknown characters are reported and rejected")
    void unknownCharTest() {
        BpeTokenizer bpe = BpeTokenizer.train(STORY, 100);
        assertEquals(Set.of('z', '#', '1'), bpe.unknownChars("zebra #1"));
        assertThrows(IllegalArgumentException.class, () -> bpe.encode("zebra"));
        assertThrows(IllegalArgumentException.class, () -> bpe.decode(new int[] {bpe.vocabSize()}));
    }

    @Test
    @DisplayName("<|endoftext|> is one atomic token and is never merged")
    void endOfTextTest() {
        String text = ("Lily was happy." + BpeTokenizer.END_OF_TEXT + "Once upon a time\n").repeat(4);
        BpeTokenizer bpe = BpeTokenizer.train(text, 60);
        assertTrue(bpe.hasEndOfText());
        int eot = bpe.baseChars().length();
        assertEquals(BpeTokenizer.END_OF_TEXT, bpe.tokenText(eot));

        int[] ids = bpe.encode("happy." + BpeTokenizer.END_OF_TEXT + "Once");
        assertEquals(1, java.util.Arrays.stream(ids).filter(id -> id == eot).count());
        assertEquals("happy." + BpeTokenizer.END_OF_TEXT + "Once", bpe.decode(ids));
        // '<', '|' and '>' only appear inside the marker, so they are not base chars
        assertEquals(Set.of('<'), bpe.unknownChars("a < e"));
        for (BpeTokenizer.Pair pair : bpe.merges()) {
            assertNotEquals(eot, pair.first());
            assertNotEquals(eot, pair.second());
        }
    }

    @Test
    @DisplayName("A tokenizer without <|endoftext|> rejects it")
    void noEndOfTextTest() {
        BpeTokenizer bpe = BpeTokenizer.train(STORY, 100);
        assertFalse(bpe.hasEndOfText());
        assertThrows(IllegalArgumentException.class, () -> bpe.encode("a" + BpeTokenizer.END_OF_TEXT + "b"));
    }

    @Test
    @DisplayName("Same text and vocab size give identical merges; rebuilding from merges gives identical ids")
    void deterministicTest() {
        BpeTokenizer first = BpeTokenizer.train(STORY, 150);
        BpeTokenizer second = BpeTokenizer.train(STORY, 150);
        assertEquals(first.merges(), second.merges());
        BpeTokenizer rebuilt = new BpeTokenizer(first.baseChars(), first.hasEndOfText(), first.merges());
        assertArrayEquals(first.encode(STORY), rebuilt.encode(STORY));
    }

    @Test
    @DisplayName("Merges learned from a sample still encode every char of the full text")
    void sampleTest() {
        String text = STORY + "\nZebras jump quickly!\n";
        BpeTokenizer bpe = BpeTokenizer.train(text, 120, 50);
        assertEquals(text, bpe.decode(bpe.encode(text)));
    }

    @Test
    @DisplayName("Invalid merge lists are rejected")
    void invalidMergesTest() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new BpeTokenizer("ab", false, List.of(new BpeTokenizer.Pair(0, 2))));
        assertThrows(
                IllegalArgumentException.class,
                () -> new BpeTokenizer("ab", false, List.of(new BpeTokenizer.Pair(0, 1), new BpeTokenizer.Pair(0, 1))));
        assertThrows(IllegalArgumentException.class, () -> new BpeTokenizer("aa", false, List.of()));
        // id 2 is <|endoftext|> here and may not be merged
        assertThrows(
                IllegalArgumentException.class,
                () -> new BpeTokenizer("ab", true, List.of(new BpeTokenizer.Pair(2, 0))));
    }
}
