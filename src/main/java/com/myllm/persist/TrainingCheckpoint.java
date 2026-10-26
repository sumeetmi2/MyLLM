package com.myllm.persist;

import com.myllm.optimizer.AdamOptimizer;
import com.myllm.tokenize.Tokenizer;
import com.myllm.transformer.TransformerModel;
import java.util.Random;

/**
 * Everything the next training step depends on, taken between two steps (no forward or backward caches are live
 * then).
 *
 * <p>{@code corpusTokens} and {@code corpusCrc} fingerprint the encoded corpus, so a resume refuses to continue on
 * different data.
 */
public record TrainingCheckpoint(
        RunSpec run,
        long corpusTokens,
        long corpusCrc,
        TransformerModel model,
        Tokenizer tokenizer,
        AdamOptimizer.State adam,
        double beta1,
        double beta2,
        double eps,
        int completedSteps,
        Random trainerRandom,
        Random generatorRandom) {}
