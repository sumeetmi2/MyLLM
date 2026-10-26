package com.myllm.persist;

import static org.junit.jupiter.api.Assertions.*;

import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.optimizer.AdamOptimizer;
import com.myllm.optimizer.WarmupCosineScheduler;
import com.myllm.tokenize.CharTokenizer;
import com.myllm.tokenize.Tokenizer;
import com.myllm.trainer.BatchedTrainerLoop;
import com.myllm.trainer.ChatExample;
import com.myllm.trainer.TrainerLoop;
import com.myllm.transformer.TransformerModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class CheckpointIOTest {

    private static final String TEXT = "the cat sat on the mat. the dog sat on the log. ".repeat(6);
    private static final int T = 8;
    private static final int TOTAL = 10;
    private static final int SPLIT = 4;

    @TempDir
    Path tempDir;

    private final CharTokenizer tokenizer = new CharTokenizer(TEXT);
    private final int[] ids = tokenizer.encode(TEXT);

    private TransformerModel newModel() {
        return new TransformerModel(tokenizer.vocabSize(), T, 8, 2, 16, 2, new Random(42));
    }

    private static WarmupCosineScheduler scheduler() {
        return new WarmupCosineScheduler(0.01, 0.001, 2, TOTAL);
    }

    private static RunSpec run(int batch, int accum, boolean sft, int loraRank) {
        return new RunSpec(
                RunSpec.Mode.PRETRAIN,
                "corpus.txt",
                "",
                TOTAL,
                0.01,
                0.001,
                2,
                batch,
                accum,
                0,
                sft,
                loraRank,
                2.0 * loraRank,
                loraRank > 0,
                "t",
                100);
    }

    private List<ChatExample> examples() {
        List<ChatExample> examples = new ArrayList<>();
        Random r = new Random(3);
        for (int i = 0; i < 6; i++) {
            int[] example = r.ints(5 + i % 4, 0, tokenizer.vocabSize()).toArray(); // input = length - 1 <= T
            examples.add(new ChatExample(example, 2 + i % 3));
        }
        return examples;
    }

    // builds a trainer of the requested kind on the given model / optimizer / random, and runs `steps` steps
    private interface Trainer {
        void train(int steps);

        int completedSteps();

        void restoreProgress(int steps);
    }

    private Trainer trainer(
            TransformerModel model, AdamOptimizer adam, Random random, int batch, int accum, boolean sft) {
        if (batch == 1 && accum == 1) {
            TrainerLoop loop = sft
                    ? new TrainerLoop(model, new CrossEntropyLoss(), adam, examples(), random, scheduler())
                    : new TrainerLoop(model, new CrossEntropyLoss(), adam, ids, T, random, scheduler());
            return new Trainer() {
                @Override
                public void train(int steps) {
                    loop.train(steps, 100);
                }

                @Override
                public int completedSteps() {
                    return loop.completedSteps();
                }

                @Override
                public void restoreProgress(int steps) {
                    loop.restoreProgress(steps);
                }
            };
        }
        BatchedTrainerLoop loop = sft
                ? new BatchedTrainerLoop(
                        model, new CrossEntropyLoss(), adam, examples(), batch, accum, random, scheduler())
                : new BatchedTrainerLoop(
                        model, new CrossEntropyLoss(), adam, ids, T, batch, accum, random, scheduler());
        return new Trainer() {
            @Override
            public void train(int steps) {
                loop.train(steps, 100);
            }

            @Override
            public int completedSteps() {
                return loop.completedSteps();
            }

            @Override
            public void restoreProgress(int steps) {
                loop.restoreProgress(steps);
            }
        };
    }

    // trains TOTAL steps straight, and separately SPLIT steps + checkpoint file + fresh objects + the rest.
    // the two must end with identical weights and optimizer state
    private void assertResumeMatches(int batch, int accum, boolean sft, int loraRank) throws IOException {
        TransformerModel straight = newModel();
        if (loraRank > 0) {
            straight.enableLora(loraRank, 2.0 * loraRank, true, new Random(7));
        }
        AdamOptimizer straightAdam = new AdamOptimizer(0.01);
        trainer(straight, straightAdam, new Random(42), batch, accum, sft).train(TOTAL);

        TransformerModel first = newModel();
        if (loraRank > 0) {
            first.enableLora(loraRank, 2.0 * loraRank, true, new Random(7));
        }
        AdamOptimizer firstAdam = new AdamOptimizer(0.01);
        Random firstRandom = new Random(42);
        Trainer firstTrainer = trainer(first, firstAdam, firstRandom, batch, accum, sft);
        firstTrainer.train(SPLIT);
        Path file = tempDir.resolve("run.ckpt");
        CheckpointIO.save(
                file,
                new TrainingCheckpoint(
                        run(batch, accum, sft, loraRank),
                        ids.length,
                        CheckpointIO.corpusCrc(ids),
                        first,
                        tokenizer,
                        firstAdam.exportState(),
                        0.9,
                        0.999,
                        1e-8,
                        firstTrainer.completedSteps(),
                        firstRandom,
                        new Random(1)));

        TrainingCheckpoint loaded = CheckpointIO.load(file);
        AdamOptimizer resumedAdam = new AdamOptimizer(0.01);
        resumedAdam.importState(loaded.adam());
        Trainer resumed = trainer(loaded.model(), resumedAdam, loaded.trainerRandom(), batch, accum, sft);
        resumed.restoreProgress(loaded.completedSteps());
        resumed.train(TOTAL - SPLIT);

        assertEquals(straight.parameters(), loaded.model().parameters());
        if (loraRank > 0) {
            assertEquals(straight.loraParameters(), loaded.model().loraParameters());
        }
        AdamOptimizer.State a = straightAdam.exportState();
        AdamOptimizer.State b = resumedAdam.exportState();
        assertEquals(a.t(), b.t());
        for (int p = 0; p < a.m().size(); p++) {
            assertArrayEquals(a.m().get(p), b.m().get(p));
            assertArrayEquals(a.v().get(p), b.v().get(p));
        }
    }

    @Test
    @DisplayName("Resume equals uninterrupted: simple TrainerLoop pretraining")
    void resumeSimpleTest() throws IOException {
        assertResumeMatches(1, 1, false, 0);
    }

    @Test
    @DisplayName("Resume equals uninterrupted: batched with accumulation")
    void resumeBatchedTest() throws IOException {
        assertResumeMatches(2, 2, false, 0);
    }

    @Test
    @DisplayName("Resume equals uninterrupted: LoRA on chat examples, simple and batched")
    void resumeLoraSftTest() throws IOException {
        assertResumeMatches(1, 1, true, 2);
        assertResumeMatches(2, 1, true, 2);
    }

    @Test
    @DisplayName("Save then load restores every field exactly, including both random generators' positions")
    void roundTripTest() throws IOException {
        TransformerModel model = newModel();
        AdamOptimizer adam = new AdamOptimizer(0.01);
        new TrainerLoop(model, new CrossEntropyLoss(), adam, ids, T, new Random(42)).train(3, 100);
        Random trainerRandom = new Random(5);
        Random generatorRandom = new Random(6);
        trainerRandom.nextInt();
        generatorRandom.nextGaussian(); // leaves a cached second gaussian, which must survive too
        RunSpec run = run(4, 2, true, 0);
        Path file = tempDir.resolve("a.ckpt");
        CheckpointIO.save(
                file,
                new TrainingCheckpoint(
                        run,
                        123,
                        456,
                        model,
                        tokenizer,
                        adam.exportState(),
                        0.9,
                        0.999,
                        1e-8,
                        3,
                        trainerRandom,
                        generatorRandom));
        TrainingCheckpoint loaded = CheckpointIO.load(file);

        assertEquals(run, loaded.run());
        assertEquals(123, loaded.corpusTokens());
        assertEquals(456, loaded.corpusCrc());
        assertEquals(3, loaded.completedSteps());
        assertEquals(model.parameters(), loaded.model().parameters());
        assertEquals(tokenizer.vocab(), ((CharTokenizer) loaded.tokenizer()).vocab());
        assertEquals(3, loaded.adam().t());
        for (int i = 0; i < 100; i++) {
            assertEquals(trainerRandom.nextInt(), loaded.trainerRandom().nextInt());
            assertEquals(
                    generatorRandom.nextGaussian(), loaded.generatorRandom().nextGaussian());
        }
        assertFalse(Files.exists(CheckpointIO.tmpPath(file.toAbsolutePath())), "temp file left behind");
    }

    @Test
    @DisplayName("Adam state saved before any step restores as empty")
    void adamStateBeforeFirstStepTest() {
        AdamOptimizer adam = new AdamOptimizer(0.01);
        AdamOptimizer.State state = adam.exportState();
        assertEquals(0, state.t());
        assertTrue(state.m().isEmpty());
        AdamOptimizer restored = new AdamOptimizer(0.01);
        restored.importState(state);
        assertEquals(0, restored.exportState().t());
    }

    @Test
    @DisplayName("A save that fails part-way leaves the previous checkpoint intact and no temp file")
    void failedSaveKeepsOldCheckpointTest() throws IOException {
        TransformerModel model = newModel();
        Path file = tempDir.resolve("b.ckpt");
        CheckpointIO.save(
                file,
                new TrainingCheckpoint(
                        run(1, 1, false, 0),
                        1,
                        2,
                        model,
                        tokenizer,
                        new AdamOptimizer(0.01).exportState(),
                        0.9,
                        0.999,
                        1e-8,
                        7,
                        new Random(1),
                        new Random(2)));
        // a tokenizer that doesn't fit the model fails inside the model section, after the header is written
        Tokenizer wrong = new CharTokenizer("ab");
        assertThrows(
                IllegalArgumentException.class,
                () -> CheckpointIO.save(
                        file,
                        new TrainingCheckpoint(
                                run(1, 1, false, 0),
                                1,
                                2,
                                model,
                                wrong,
                                new AdamOptimizer(0.01).exportState(),
                                0.9,
                                0.999,
                                1e-8,
                                9,
                                new Random(1),
                                new Random(2))));
        assertEquals(7, CheckpointIO.load(file).completedSteps());
        assertFalse(Files.exists(CheckpointIO.tmpPath(file.toAbsolutePath())), "temp file left behind");
    }

    @Test
    @DisplayName("Truncated or foreign files fail with a clear error")
    void corruptFilesTest() throws IOException {
        Path file = tempDir.resolve("c.ckpt");
        CheckpointIO.save(
                file,
                new TrainingCheckpoint(
                        run(1, 1, false, 0),
                        1,
                        2,
                        newModel(),
                        tokenizer,
                        new AdamOptimizer(0.01).exportState(),
                        0.9,
                        0.999,
                        1e-8,
                        0,
                        new Random(1),
                        new Random(2)));
        byte[] bytes = Files.readAllBytes(file);
        Path truncated = tempDir.resolve("truncated.ckpt");
        Files.write(truncated, Arrays.copyOf(bytes, bytes.length / 2));
        IOException e = assertThrows(IOException.class, () -> CheckpointIO.load(truncated));
        assertTrue(e.getMessage().contains("truncated"), e.getMessage());

        Path foreign = tempDir.resolve("foreign.ckpt");
        Files.write(foreign, new byte[] {1, 2, 3, 4, 5, 6, 7, 8});
        e = assertThrows(IOException.class, () -> CheckpointIO.load(foreign));
        assertTrue(e.getMessage().contains("not a MyLLM checkpoint"), e.getMessage());
    }

    @Test
    @DisplayName("Corpus fingerprint changes when a single token changes")
    void corpusCrcTest() {
        int[] changed = ids.clone();
        changed[changed.length / 2] ^= 1;
        assertNotEquals(CheckpointIO.corpusCrc(ids), CheckpointIO.corpusCrc(changed));
        assertEquals(CheckpointIO.corpusCrc(ids), CheckpointIO.corpusCrc(ids.clone()));
    }
}
