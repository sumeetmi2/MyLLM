package com.myllm.corpus;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class CorpusCleanerTest {

    @Test
    @DisplayName("YAML front matter at the start of a document is removed")
    void frontMatterTest() {
        String doc = "---\ntitle: Fees\ntags: [a, b]\n---\n# Fees\nThe fee is charged once.";
        assertEquals("# Fees\nThe fee is charged once.", CorpusCleaner.cleanDocument(doc));
    }

    @Test
    @DisplayName("A --- rule later in the document is kept")
    void horizontalRuleKeptTest() {
        String doc = "# Fees\n\n---\n\nmore text";
        assertEquals(doc, CorpusCleaner.cleanDocument(doc));
    }

    @Test
    @DisplayName("Typographic symbols map to ASCII, accents are stripped, other non-ASCII is dropped")
    void asciiTest() {
        assertEquals(
                "'quoted' \"text\" - a...b -> c <= d cafe x",
                CorpusCleaner.cleanDocument("‘quoted’ “text” — a…b → c ≤ d café x🚀"));
    }

    @Test
    @DisplayName("Box drawing becomes - | +")
    void boxDrawingTest() {
        assertEquals("+-- src\n| +-- Main.java", CorpusCleaner.cleanDocument("├── src\n│ └── Main.java"));
    }

    @Test
    @DisplayName("UUIDs, hashes, long numbers and line numbers are replaced")
    void noiseTest() {
        String doc = "id 123e4567-e89b-12d3-a456-426614174000 commit 9fceb02d0ae598e95dc970b74767f19372d61af8 "
                + "amount 1000000000 see PublisherService.java:120 and Foo.kt:10-20 version 2.5 year 2024";
        assertEquals(
                "id <id> commit <id> amount <num> see PublisherService.java and Foo.kt version 2.5 year 2024",
                CorpusCleaner.cleanDocument(doc));
    }

    @Test
    @DisplayName("Whitespace: tabs, trailing spaces, inner runs and blank-line runs collapse; indentation kept")
    void whitespaceTest() {
        String doc = "| a    | b |   \n\n\n\n    indented  code\n\tTabbed";
        assertEquals("| a | b |\n\n    indented code\n    Tabbed", CorpusCleaner.cleanDocument(doc));
    }

    @Test
    @DisplayName("Ordinary words are never removed")
    void wordsKeptTest() {
        String doc = "The fee is applied to the order and the refund is then settled.";
        assertEquals(doc, CorpusCleaner.cleanDocument(doc));
    }

    @Test
    @DisplayName("Long duplicate paragraphs across documents are kept once; short ones may repeat")
    void dedupeTest() {
        String boilerplate = "This document is maintained by the pricing team. Contact them for changes.";
        String cleaned = CorpusCleaner.clean(List.of(
                "# One\n\n" + boilerplate + "\n\nfirst body", "# Two\n\n" + boilerplate + "\n\nsecond body\n\n# One"));
        assertEquals(1, cleaned.split(boilerplate, -1).length - 1);
        assertEquals(2, cleaned.split("# One", -1).length - 1);
        assertTrue(cleaned.contains("first body") && cleaned.contains("second body"));
    }

    @Test
    @DisplayName("Empty documents produce no output")
    void emptyTest() {
        assertEquals("", CorpusCleaner.clean(List.of("", "---\na: b\n---\n", "   \n\n")));
    }
}
