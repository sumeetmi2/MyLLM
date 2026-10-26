package com.myllm.trainer;

/**
 * One chat example for supervised fine-tuning: prompt tokens, then answer tokens ending with
 * {@code <|endoftext|>}.
 *
 * <p>{@code answerStart} is the index of the first answer token; only answer tokens are graded by the loss.
 */
public record ChatExample(int[] ids, int answerStart) {

    public ChatExample {
        if (answerStart <= 0 || answerStart >= ids.length) {
            throw new IllegalArgumentException(
                    "answerStart " + answerStart + " must be inside (0, " + ids.length + ")");
        }
    }
}
