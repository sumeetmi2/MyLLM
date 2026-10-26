package com.myllm.persist;

import com.myllm.dao.Matrix;
import com.myllm.tokenize.BpeTokenizer;
import com.myllm.tokenize.CharTokenizer;
import com.myllm.tokenize.Tokenizer;
import com.myllm.transformer.TransformerModel;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Binary save and load of a trained model plus its tokenizer.
 *
 * <p>Layout: MAGIC, VERSION, config ints, tokenizer, parameter count, then per parameter: rows, cols, values.
 * <ul>
 *   <li>Version 1 (char tokenizer only): tokenizer = vocab length + chars.
 *   <li>Version 2: tokenizer = type byte, then char vocab, or BPE base chars + special flag + merge pairs.
 *   <li>Version 3: same as 2, but values are float32 instead of float64. Versions 1 and 2 still load, rounded to
 *       float.
 * </ul>
 * Parameters are written in {@code TransformerModel.parameters()} order and read back into a fresh model in the
 * same order.
 */
public class ModelIO {

    private static final int MAGIC = 0x4D594C4D; // "MYLM"
    private static final int VERSION = 3;
    private static final byte TOKENIZER_CHAR = 0;
    private static final byte TOKENIZER_BPE = 1;
    // strings are vocabularies and file paths; 1M chars is far above any real one
    private static final int MAX_STRING_CHARS = 1 << 20;

    public record LoadedModel(TransformerModel model, Tokenizer tokenizer) {}

    public static void save(Path path, TransformerModel model, Tokenizer tokenizer) throws IOException {
        // parameters() holds only the base weights; saving now would silently drop everything the adapters learned
        if (model.isLoraEnabled()) {
            throw new IllegalStateException("model has LoRA adapters; merge them into the base weights before saving");
        }
        // checked again in writeBody, but here it fails before an empty file is created
        if (tokenizer.vocabSize() != model.getConfig().vocabSize()) {
            throw new IllegalArgumentException("tokenizer vocab size " + tokenizer.vocabSize() + " != model vocab size "
                    + model.getConfig().vocabSize());
        }
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            writeBody(out, model, tokenizer);
        }
    }

    // everything after MAGIC and VERSION: config, tokenizer, base weights. Also the model section of a checkpoint,
    // which is why it writes the base weights even with LoRA on (the checkpoint stores the adapters itself)
    static void writeBody(DataOutputStream out, TransformerModel model, Tokenizer tokenizer) throws IOException {
        TransformerModel.ModelConfig config = model.getConfig();
        if (tokenizer.vocabSize() != config.vocabSize()) {
            throw new IllegalArgumentException(
                    "tokenizer vocab size " + tokenizer.vocabSize() + " != model vocab size " + config.vocabSize());
        }
        out.writeInt(config.vocabSize());
        out.writeInt(config.T_max());
        out.writeInt(config.d_Model());
        out.writeInt(config.heads());
        out.writeInt(config.d_ff());
        out.writeInt(config.noOfDecoderBlocks());
        writeTokenizer(out, tokenizer);
        writeMatrices(out, model.parameters());
    }

    public static LoadedModel load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            if (in.readInt() != MAGIC) {
                throw new IOException(path + " is not a MyLLM model file");
            }
            int version = in.readInt();
            if (version < 1 || version > VERSION) {
                throw new IOException("unsupported model file version " + version + ", expected 1 to " + VERSION);
            }
            return readBody(in, version);
        }
    }

    static LoadedModel readBody(DataInputStream in, int version) throws IOException {
        TransformerModel.ModelConfig config = new TransformerModel.ModelConfig(
                in.readInt(), in.readInt(), in.readInt(), in.readInt(), in.readInt(), in.readInt());
        // version 1 files always used the char tokenizer and had no type byte
        Tokenizer tokenizer = version == 1 ? new CharTokenizer(readString(in)) : readTokenizer(in);
        if (tokenizer.vocabSize() != config.vocabSize()) {
            throw new IOException(
                    "tokenizer has " + tokenizer.vocabSize() + " tokens, model expects " + config.vocabSize());
        }
        // fresh model has the right shapes;
        // its random weights are overwritten in place below, so the seed doesn't matter
        TransformerModel model = new TransformerModel(config, new Random(0));
        readMatricesInto(in, model.parameters(), version >= 3);
        return new LoadedModel(model, tokenizer);
    }

    // count, then per matrix: rows, cols, float32 values
    static void writeMatrices(DataOutputStream out, List<Matrix> matrices) throws IOException {
        out.writeInt(matrices.size());
        for (Matrix matrix : matrices) {
            out.writeInt(matrix.rows());
            out.writeInt(matrix.cols());
            writeFloats(out, matrix.data());
        }
    }

    // overwrites each matrix in place; shapes in the file must match. floats = false reads the old float64 layout
    static void readMatricesInto(DataInputStream in, List<Matrix> matrices, boolean floats) throws IOException {
        int storedCount = in.readInt();
        if (storedCount != matrices.size()) {
            throw new IOException("file has " + storedCount + " parameters, model expects " + matrices.size());
        }
        for (int p = 0; p < matrices.size(); p++) {
            Matrix matrix = matrices.get(p);
            int rows = in.readInt();
            int cols = in.readInt();
            if (rows != matrix.rows() || cols != matrix.cols()) {
                throw new IOException("parameter " + p + " is " + rows + "x" + cols + " in file, model expects "
                        + matrix.rows() + "x" + matrix.cols());
            }
            float[] values = matrix.data();
            if (floats) {
                readFloats(in, values);
            } else {
                // older files stored doubles
                for (int k = 0; k < values.length; k++) {
                    values[k] = (float) in.readDouble();
                }
            }
        }
    }

    // one bulk write; ByteBuffer is big-endian like the rest of the DataOutputStream
    static void writeFloats(DataOutputStream out, float[] values) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate(Float.BYTES * values.length);
        bytes.asFloatBuffer().put(values);
        out.write(bytes.array());
    }

    static void readFloats(DataInputStream in, float[] values) throws IOException {
        byte[] bytes = new byte[Float.BYTES * values.length];
        in.readFully(bytes);
        ByteBuffer.wrap(bytes).asFloatBuffer().get(values);
    }

    private static void writeTokenizer(DataOutputStream out, Tokenizer tokenizer) throws IOException {
        if (tokenizer instanceof CharTokenizer chars) {
            out.writeByte(TOKENIZER_CHAR);
            writeString(out, chars.vocab());
        } else if (tokenizer instanceof BpeTokenizer bpe) {
            out.writeByte(TOKENIZER_BPE);
            writeString(out, bpe.baseChars());
            out.writeBoolean(bpe.hasEndOfText());
            out.writeInt(bpe.merges().size());
            for (BpeTokenizer.Pair pair : bpe.merges()) {
                out.writeInt(pair.first());
                out.writeInt(pair.second());
            }
        } else {
            throw new IllegalArgumentException(
                    "can't save tokenizer type " + tokenizer.getClass().getName());
        }
    }

    private static Tokenizer readTokenizer(DataInputStream in) throws IOException {
        byte type = in.readByte();
        if (type == TOKENIZER_CHAR) {
            return new CharTokenizer(readString(in));
        }
        if (type != TOKENIZER_BPE) {
            throw new IOException("unknown tokenizer type " + type);
        }
        String baseChars = readString(in);
        boolean hasEndOfText = in.readBoolean();
        int count = in.readInt();
        List<BpeTokenizer.Pair> merges = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            merges.add(new BpeTokenizer.Pair(in.readInt(), in.readInt()));
        }
        try {
            return new BpeTokenizer(baseChars, hasEndOfText, merges);
        } catch (IllegalArgumentException e) {
            throw new IOException("corrupt BPE tokenizer: " + e.getMessage(), e);
        }
    }

    // length + raw chars instead of writeUTF, which caps at 64KB
    static void writeString(DataOutputStream out, String s) throws IOException {
        out.writeInt(s.length());
        out.writeChars(s);
    }

    static String readString(DataInputStream in) throws IOException {
        int length = in.readInt();
        // a corrupt length must fail here, not allocate up to 4GB first
        if (length < 0 || length > MAX_STRING_CHARS) {
            throw new IOException("corrupt file: string length " + length);
        }
        char[] chars = new char[length];
        for (int i = 0; i < chars.length; i++) {
            chars[i] = in.readChar();
        }
        return new String(chars);
    }
}
