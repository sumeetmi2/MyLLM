package com.myllm.persist;

/**
 * The settings of a training run, stored in its checkpoints so {@code resume <ckpt>} can rebuild the run with no
 * arguments. {@code savePath = ""} means no model is saved at the end.
 */
public record RunSpec(
        Mode mode,
        String corpusPath,
        String savePath,
        int totalSteps,
        double peakLr,
        double minLr,
        int warmupSteps,
        int batch,
        int accum,
        int bpeVocab,
        boolean sft,
        int loraRank,
        double loraAlpha,
        boolean loraAll,
        String samplePrompt,
        int checkpointEvery) {

    public enum Mode {
        PRETRAIN,
        FINETUNE
    }

    public boolean lora() {
        return loraRank > 0;
    }
}
