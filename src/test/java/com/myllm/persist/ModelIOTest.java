package com.myllm.persist;

import static com.myllm.utils.TestUtils.assertMatrixEquals;
import static org.junit.jupiter.api.Assertions.*;

import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.dao.Matrix;
import com.myllm.generator.Generator;
import com.myllm.optimizer.SGDOptimizer;
import com.myllm.tokenize.BpeTokenizer;
import com.myllm.tokenize.CharTokenizer;
import com.myllm.trainer.TrainerLoop;
import com.myllm.transformer.TransformerModel;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class ModelIOTest {

    private static final String TEXT = "the cat sat on the mat. the dog sat on the log. ".repeat(4);
    private static final int T_MAX = 8;

    @TempDir
    Path tempDir;

    private TransformerModel trainedModel(CharTokenizer tokenizer) {
        TransformerModel model = new TransformerModel(tokenizer.vocabSize(), T_MAX, 8, 2, 16, 2, new Random(42));
        new TrainerLoop(
                        model,
                        new CrossEntropyLoss(),
                        new SGDOptimizer(0.1),
                        tokenizer.encode(TEXT),
                        T_MAX,
                        new Random(42))
                .train(50, 50);
        return model;
    }

    @Test
    @DisplayName("Tokenizer rebuilt from vocab() gives the same ids")
    void vocabRoundTripTest() {
        CharTokenizer original = new CharTokenizer(TEXT);
        CharTokenizer rebuilt = new CharTokenizer(original.vocab());
        assertEquals(original.vocabSize(), rebuilt.vocabSize());
        assertArrayEquals(original.encode(TEXT), rebuilt.encode(TEXT));
    }

    @Test
    @DisplayName("Saved then loaded model produces identical logits and identical generated text")
    void saveLoadRoundTripTest() throws IOException {
        CharTokenizer tokenizer = new CharTokenizer(TEXT);
        TransformerModel model = trainedModel(tokenizer);
        Path file = tempDir.resolve("models/model.bin");

        ModelIO.save(file, model, tokenizer);
        ModelIO.LoadedModel loaded = ModelIO.load(file);

        assertEquals(model.getConfig(), loaded.model().getConfig());
        CharTokenizer loadedTokenizer = assertInstanceOf(CharTokenizer.class, loaded.tokenizer());
        assertEquals(tokenizer.vocab(), loadedTokenizer.vocab());
        int[] ids = tokenizer.encode("the cat");
        // float32 values are written as their raw bits, so weights and logits must match bit for bit
        assertEquals(model.parameters(), loaded.model().parameters());
        assertMatrixEquals(model.forwardPass(ids).toArray(), loaded.model().forwardPass(ids), 0.0);
        assertEquals(
                new Generator(model, tokenizer).generate("t", 30),
                new Generator(loaded.model(), loaded.tokenizer()).generate("t", 30));
    }

    @Test
    @DisplayName("Loading a file that is not a model fails with a clear error")
    void wrongMagicTest() throws IOException {
        Path file = tempDir.resolve("not-a-model.bin");
        Files.writeString(file, "hello, definitely not a model");
        IOException e = assertThrows(IOException.class, () -> ModelIO.load(file));
        assertTrue(e.getMessage().contains("not a MyLLM model"));
    }

    @Test
    @DisplayName("Loading a file with an unknown version fails")
    void wrongVersionTest() throws IOException {
        Path file = tempDir.resolve("future.bin");
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
            out.writeInt(0x4D594C4D);
            out.writeInt(99);
        }
        IOException e = assertThrows(IOException.class, () -> ModelIO.load(file));
        assertTrue(e.getMessage().contains("version"));
    }

    @Test
    @DisplayName("BPE model round trip: same tokenizer ids and identical logits")
    void bpeRoundTripTest() throws IOException {
        String text = ("Lily was happy." + BpeTokenizer.END_OF_TEXT + "Once upon a time, Tom saw a dog.\n").repeat(6);
        BpeTokenizer tokenizer = BpeTokenizer.train(text, 50);
        TransformerModel model = new TransformerModel(tokenizer.vocabSize(), T_MAX, 8, 2, 16, 1, new Random(42));
        Path file = tempDir.resolve("bpe.bin");

        ModelIO.save(file, model, tokenizer);
        ModelIO.LoadedModel loaded = ModelIO.load(file);

        BpeTokenizer loadedTokenizer = assertInstanceOf(BpeTokenizer.class, loaded.tokenizer());
        assertArrayEquals(tokenizer.encode(text), loadedTokenizer.encode(text));
        assertEquals(tokenizer.hasEndOfText(), loadedTokenizer.hasEndOfText());
        int[] ids = tokenizer.encode("Once upon a time");
        assertMatrixEquals(model.forwardPass(ids).toArray(), loaded.model().forwardPass(ids), 0.0);
    }

    @Test
    @DisplayName("Version 1 files (char tokenizer, no type byte) still load")
    void loadsVersion1Test() throws IOException {
        CharTokenizer tokenizer = new CharTokenizer(TEXT);
        TransformerModel model = new TransformerModel(tokenizer.vocabSize(), T_MAX, 8, 2, 16, 1, new Random(42));
        Path file = tempDir.resolve("v1.bin");
        TransformerModel.ModelConfig config = model.getConfig();
        // the exact layout ModelIO wrote before BPE support
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
            out.writeInt(0x4D594C4D);
            out.writeInt(1);
            for (int value : new int[] {
                config.vocabSize(),
                config.T_max(),
                config.d_Model(),
                config.heads(),
                config.d_ff(),
                config.noOfDecoderBlocks()
            }) {
                out.writeInt(value);
            }
            out.writeInt(tokenizer.vocab().length());
            out.writeChars(tokenizer.vocab());
            out.writeInt(model.parameters().size());
            for (Matrix parameter : model.parameters()) {
                out.writeInt(parameter.rows());
                out.writeInt(parameter.cols());
                for (double value : parameter.data()) {
                    out.writeDouble(value);
                }
            }
        }

        ModelIO.LoadedModel loaded = ModelIO.load(file);
        CharTokenizer loadedTokenizer = assertInstanceOf(CharTokenizer.class, loaded.tokenizer());
        assertEquals(tokenizer.vocab(), loadedTokenizer.vocab());
        int[] ids = tokenizer.encode("the cat");
        assertMatrixEquals(model.forwardPass(ids).toArray(), loaded.model().forwardPass(ids), 0.0);
    }

    @Test
    @DisplayName("Version 2 files (float64 values) still load, each value rounded to float")
    void loadsVersion2Test() throws IOException {
        CharTokenizer tokenizer = new CharTokenizer(TEXT);
        TransformerModel model = new TransformerModel(tokenizer.vocabSize(), T_MAX, 8, 2, 16, 1, new Random(42));
        Path file = tempDir.resolve("v2.bin");
        TransformerModel.ModelConfig config = model.getConfig();
        // the exact layout ModelIO wrote before switching to float32: tokenizer type byte, then doubles.
        // the tiny offset makes each stored double one that float can't hold exactly, so the rounding is exercised
        double offset = 1e-10;
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
            out.writeInt(0x4D594C4D);
            out.writeInt(2);
            for (int value : new int[] {
                config.vocabSize(),
                config.T_max(),
                config.d_Model(),
                config.heads(),
                config.d_ff(),
                config.noOfDecoderBlocks()
            }) {
                out.writeInt(value);
            }
            out.writeByte(0);
            out.writeInt(tokenizer.vocab().length());
            out.writeChars(tokenizer.vocab());
            out.writeInt(model.parameters().size());
            for (Matrix parameter : model.parameters()) {
                out.writeInt(parameter.rows());
                out.writeInt(parameter.cols());
                for (float value : parameter.data()) {
                    out.writeDouble(value + offset);
                }
            }
        }

        ModelIO.LoadedModel loaded = ModelIO.load(file);
        for (int p = 0; p < model.parameters().size(); p++) {
            float[] original = model.parameters().get(p).data();
            float[] read = loaded.model().parameters().get(p).data();
            for (int k = 0; k < original.length; k++) {
                assertEquals((float) (original[k] + offset), read[k], "parameter " + p + " value " + k);
            }
        }
    }
}
