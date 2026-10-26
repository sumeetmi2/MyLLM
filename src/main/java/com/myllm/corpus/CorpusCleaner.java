package com.myllm.corpus;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Cleans raw documents before training.
 *
 * <p>Removes noise the model would waste capacity on: YAML front matter, ids and hashes, long numbers, line
 * numbers, non-ASCII symbols, messy whitespace and duplicate paragraphs. Ordinary words are never removed, so the
 * model still learns normal grammar.
 */
public final class CorpusCleaner {

    // paragraphs shorter than this (headings, "---", table separators) may repeat legitimately
    static final int MIN_DEDUPE_LENGTH = 40;

    private static final Pattern FRONT_MATTER = Pattern.compile("\\A---\\n.*?\\n---(\\n|\\z)", Pattern.DOTALL);
    private static final Pattern UUID =
            Pattern.compile("\\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\\b");
    // 12+ hex chars containing both a digit and a letter: commit ids, hashes
    private static final Pattern HEX_ID =
            Pattern.compile("\\b(?=[0-9a-fA-F]*\\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{12,}\\b");
    // "Foo.java:123" or "Foo.java:12-40" -> "Foo.java"
    private static final Pattern LINE_NUMBER = Pattern.compile("(\\.[A-Za-z]{1,5}):\\d+(-\\d+)?\\b");
    private static final Pattern LONG_NUMBER = Pattern.compile("\\d{6,}");
    private static final Pattern TRAILING_SPACES = Pattern.compile("[ ]+$", Pattern.MULTILINE);
    // runs of spaces after a non-space; leading indentation is kept
    private static final Pattern INNER_SPACES = Pattern.compile("(?<=\\S) {2,}");
    private static final Pattern BLANK_LINES = Pattern.compile("\\n{3,}");
    private static final Pattern COMBINING_MARKS = Pattern.compile("\\p{M}+");

    private CorpusCleaner() {}

    // cleans each document, joins them with a blank line, then drops duplicate paragraphs across all of them
    public static String clean(List<String> documents) {
        StringBuilder corpus = new StringBuilder();
        for (String document : documents) {
            String cleaned = cleanDocument(document);
            if (!cleaned.isEmpty()) {
                corpus.append(cleaned).append("\n\n");
            }
        }
        return dedupeParagraphs(corpus.toString());
    }

    public static String cleanDocument(String document) {
        String text = document.replace("﻿", "").replace("\r\n", "\n").replace('\r', '\n');
        text = FRONT_MATTER.matcher(text).replaceFirst("");
        text = toAscii(text);
        // UUID before HEX_ID: UUID segments are hex too
        text = UUID.matcher(text).replaceAll("<id>");
        text = HEX_ID.matcher(text).replaceAll("<id>");
        text = LINE_NUMBER.matcher(text).replaceAll("$1");
        text = LONG_NUMBER.matcher(text).replaceAll("<num>");
        text = text.replace("\t", "    ");
        text = TRAILING_SPACES.matcher(text).replaceAll("");
        text = INNER_SPACES.matcher(text).replaceAll(" ");
        text = BLANK_LINES.matcher(text).replaceAll("\n\n");
        return text.strip();
    }

    // maps common typographic symbols to ASCII, strips accents, drops anything else outside printable ASCII
    static String toAscii(String text) {
        StringBuilder mapped = new StringBuilder(text.length());
        text.codePoints().forEach(c -> mapped.append(asciiFor(c)));
        // NFKD splits "é" into "e" + accent mark, the mark is then removed
        String decomposed = COMBINING_MARKS
                .matcher(Normalizer.normalize(mapped, Normalizer.Form.NFKD))
                .replaceAll("");
        StringBuilder result = new StringBuilder(decomposed.length());
        for (int i = 0; i < decomposed.length(); i++) {
            char c = decomposed.charAt(i);
            if (c == '\n' || c == '\t' || (c >= ' ' && c <= '~')) {
                result.append(c);
            }
        }
        return result.toString();
    }

    private static String asciiFor(int c) {
        return switch (c) {
            case '‘', '’', '‚', '′' -> "'";
            case '“', '”', '„', '″' -> "\"";
            case '‐', '‑', '‒', '–', '—', '―', '−' -> "-";
            case '…' -> "...";
            case '→', '⇒', '➜', '➡' -> "->";
            case '←', '⇐' -> "<-";
            case '↔', '⇔' -> "<->";
            case '•', '·', '●', '▪', '◦' -> "-";
            case '×' -> "x";
            case '≤' -> "<=";
            case '≥' -> ">=";
            case '≠' -> "!=";
            case '≈' -> "~";
            case ' ', ' ', ' ', ' ', ' ' -> " ";
            case '​', '‌', '‍', '️' -> "";
            case '✅', '✓', '✔' -> "yes";
            case '❌', '✗', '✘' -> "no";
            default -> boxDrawing(c);
        };
    }

    // tree / table drawing characters: lines become - or |, corners and joints become +
    private static String boxDrawing(int c) {
        if (c < 0x2500 || c > 0x257F) {
            return Character.toString(c);
        }
        return switch (c) {
            case '─', '━', '═' -> "-";
            case '│', '┃', '║' -> "|";
            default -> "+";
        };
    }

    // keeps the first copy of each long paragraph (split on blank lines)
    static String dedupeParagraphs(String text) {
        Set<String> seen = new HashSet<>();
        List<String> kept = new ArrayList<>();
        for (String paragraph : text.split("\\n\\n+")) {
            String trimmed = paragraph.strip();
            if (trimmed.isEmpty()) {
                continue;
            }
            if (trimmed.length() >= MIN_DEDUPE_LENGTH && !seen.add(trimmed)) {
                continue;
            }
            kept.add(trimmed);
        }
        return kept.isEmpty() ? "" : String.join("\n\n", kept) + "\n";
    }
}
