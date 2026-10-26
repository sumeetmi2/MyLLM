package com.myllm.trainer;

import com.myllm.tokenize.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Splits chat text ({@code "User: ...\nAssistant: ...<|endoftext|>"} repeated) into {@code ChatExample}s.
 */
public final class ChatExamples {

    public static final String END_OF_TEXT = "<|endoftext|>";
    public static final String ASSISTANT_MARKER = "\nAssistant:";

    public record Result(List<ChatExample> examples, int tooLong, int noMarker) {}

    private ChatExamples() {}

    // prompt and answer are encoded separately and concatenated: chat encodes the prompt on its own and then generates,
    // so training must see the same token boundary after "Assistant:" (encoding the whole piece could merge across it).
    // Examples longer than tMax + 1 tokens are dropped, not truncated:
    // a cut-off answer never shows the model how to stop
    public static Result build(String text, Tokenizer tokenizer, int tMax) {
        List<ChatExample> examples = new ArrayList<>();
        int tooLong = 0;
        int noMarker = 0;
        for (String piece : text.split(Pattern.quote(END_OF_TEXT))) {
            String trimmed = piece.strip();
            if (trimmed.isEmpty()) {
                continue;
            }
            int marker = trimmed.indexOf(ASSISTANT_MARKER);
            if (marker < 0) {
                noMarker++;
                continue;
            }
            int answerFrom = marker + ASSISTANT_MARKER.length();
            int[] prompt = tokenizer.encode(trimmed.substring(0, answerFrom));
            int[] answer = tokenizer.encode(trimmed.substring(answerFrom) + "\n" + END_OF_TEXT);
            // a window feeds ids[0 .. n-2] as input, so n - 1 tokens must fit in tMax
            if (prompt.length + answer.length > tMax + 1) {
                tooLong++;
                continue;
            }
            int[] ids = new int[prompt.length + answer.length];
            System.arraycopy(prompt, 0, ids, 0, prompt.length);
            System.arraycopy(answer, 0, ids, prompt.length, answer.length);
            examples.add(new ChatExample(ids, prompt.length));
        }
        return new Result(examples, tooLong, noMarker);
    }
}
