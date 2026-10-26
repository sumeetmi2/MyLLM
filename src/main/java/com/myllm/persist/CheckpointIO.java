package com.myllm.persist;

import com.myllm.optimizer.AdamOptimizer;
import com.myllm.transformer.TransformerModel;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.CRC32;

/**
 * Training checkpoints: model, LoRA, Adam state, progress and RNG positions, so a killed run can resume and end
 * with exactly the model an uninterrupted run would have produced.
 *
 * <p>Layout (big-endian): MAGIC, VERSION, run spec, corpus fingerprint, ModelIO body (base weights), LoRA section,
 * Adam section, completedSteps, trainer and generator {@code Random} (Java serialization), END marker.
 *
 * <p>Writes go to {@code <path>.tmp}, are forced to disk, then atomically renamed over {@code <path>}, so a kill at
 * any moment leaves either the previous checkpoint or the new one. The previous one is also kept as
 * {@code <path>.prev}.
 */
public final class CheckpointIO {

    private static final int MAGIC = 0x4D59434B; // "MYCK"
    private static final int VERSION = 1;
    private static final int END = 0x444F4E45; // "DONE"
    private static final byte OPTIMIZER_ADAM = 0;

    private CheckpointIO() {}

    public static void save(Path path, TrainingCheckpoint checkpoint) throws IOException {
        Path target = path.toAbsolutePath();
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path tmp = tmpPath(target);
        try {
            try (FileOutputStream file = new FileOutputStream(tmp.toFile());
                    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(file, 1 << 20))) {
                write(out, checkpoint);
                out.flush();
                // the rename below must not overtake the data: force it to disk first
                file.getChannel().force(true);
            }
            keepPrevious(target);
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(tmp);
            throw e;
        }
    }

    // a hard link costs nothing even for a GB-sized file; without link support there is just no .prev
    private static void keepPrevious(Path target) throws IOException {
        if (!Files.exists(target)) {
            return;
        }
        Path previous = target.resolveSibling(target.getFileName() + ".prev");
        Files.deleteIfExists(previous);
        try {
            Files.createLink(previous, target);
        } catch (UnsupportedOperationException | IOException e) {
            // not fatal: the atomic move still protects the current checkpoint
        }
    }

    static Path tmpPath(Path target) {
        return target.resolveSibling(target.getFileName() + ".tmp");
    }

    private static void write(DataOutputStream out, TrainingCheckpoint c) throws IOException {
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        writeRun(out, c.run());
        out.writeLong(c.corpusTokens());
        out.writeLong(c.corpusCrc());
        ModelIO.writeBody(out, c.model(), c.tokenizer());

        TransformerModel model = c.model();
        out.writeBoolean(model.isLoraEnabled());
        if (model.isLoraEnabled()) {
            ModelIO.writeMatrices(out, model.loraParameters());
        }

        out.writeByte(OPTIMIZER_ADAM);
        out.writeDouble(c.beta1());
        out.writeDouble(c.beta2());
        out.writeDouble(c.eps());
        AdamOptimizer.State adam = c.adam();
        out.writeInt(adam.t());
        out.writeInt(adam.m().size());
        for (int p = 0; p < adam.m().size(); p++) {
            out.writeInt(adam.m().get(p).length);
            ModelIO.writeFloats(out, adam.m().get(p));
            ModelIO.writeFloats(out, adam.v().get(p));
        }

        out.writeInt(c.completedSteps());
        writeRandom(out, c.trainerRandom());
        writeRandom(out, c.generatorRandom());
        out.writeInt(END);
    }

    public static TrainingCheckpoint load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path), 1 << 20))) {
            if (in.readInt() != MAGIC) {
                throw new IOException(path + " is not a MyLLM checkpoint");
            }
            int version = in.readInt();
            if (version != VERSION) {
                throw new IOException("unsupported checkpoint version " + version + ", expected " + VERSION);
            }
            RunSpec run = readRun(in);
            long corpusTokens = in.readLong();
            long corpusCrc = in.readLong();
            ModelIO.LoadedModel loaded = ModelIO.readBody(in, 3);
            TransformerModel model = loaded.model();

            boolean lora = in.readBoolean();
            if (lora != run.lora()) {
                throw new IOException("checkpoint LoRA flag " + lora + " doesn't match its run settings");
            }
            if (lora) {
                // the seed doesn't matter, the stored adapter values overwrite it
                model.enableLora(run.loraRank(), run.loraAlpha(), run.loraAll(), new Random(0));
                ModelIO.readMatricesInto(in, model.loraParameters(), true);
            }

            byte optimizer = in.readByte();
            if (optimizer != OPTIMIZER_ADAM) {
                throw new IOException("unknown optimizer type " + optimizer);
            }
            double beta1 = in.readDouble();
            double beta2 = in.readDouble();
            double eps = in.readDouble();
            int t = in.readInt();
            int count = in.readInt();
            List<float[]> m = new ArrayList<>(count);
            List<float[]> v = new ArrayList<>(count);
            for (int p = 0; p < count; p++) {
                int length = in.readInt();
                float[] mp = new float[length];
                float[] vp = new float[length];
                ModelIO.readFloats(in, mp);
                ModelIO.readFloats(in, vp);
                m.add(mp);
                v.add(vp);
            }

            int completedSteps = in.readInt();
            Random trainerRandom = readRandom(in);
            Random generatorRandom = readRandom(in);
            if (in.readInt() != END) {
                throw new IOException(path + " is corrupt: end marker missing");
            }
            return new TrainingCheckpoint(
                    run,
                    corpusTokens,
                    corpusCrc,
                    model,
                    loaded.tokenizer(),
                    new AdamOptimizer.State(t, m, v),
                    beta1,
                    beta2,
                    eps,
                    completedSteps,
                    trainerRandom,
                    generatorRandom);
        } catch (EOFException e) {
            throw new IOException(path + " is truncated (an interrupted copy?)", e);
        }
    }

    // fingerprint of the encoded corpus: catches a changed file, a different cleaner or a different tokenizer
    public static long corpusCrc(int[] ids) {
        CRC32 crc = new CRC32();
        ByteBuffer buffer = ByteBuffer.allocate(4 * 65_536);
        for (int from = 0; from < ids.length; from += 65_536) {
            buffer.clear();
            buffer.asIntBuffer().put(ids, from, Math.min(65_536, ids.length - from));
            crc.update(buffer.array(), 0, 4 * Math.min(65_536, ids.length - from));
        }
        return crc.getValue();
    }

    private static void writeRun(DataOutputStream out, RunSpec r) throws IOException {
        out.writeByte(modeCode(r.mode()));
        ModelIO.writeString(out, r.corpusPath());
        ModelIO.writeString(out, r.savePath());
        out.writeInt(r.totalSteps());
        out.writeDouble(r.peakLr());
        out.writeDouble(r.minLr());
        out.writeInt(r.warmupSteps());
        out.writeInt(r.batch());
        out.writeInt(r.accum());
        out.writeInt(r.bpeVocab());
        out.writeBoolean(r.sft());
        out.writeInt(r.loraRank());
        out.writeDouble(r.loraAlpha());
        out.writeBoolean(r.loraAll());
        ModelIO.writeString(out, r.samplePrompt());
        out.writeInt(r.checkpointEvery());
    }

    // fixed codes, so reordering or adding enum constants can't change what old checkpoints mean
    private static int modeCode(RunSpec.Mode mode) {
        return switch (mode) {
            case PRETRAIN -> 0;
            case FINETUNE -> 1;
        };
    }

    private static RunSpec.Mode modeFromCode(int code) throws IOException {
        return switch (code) {
            case 0 -> RunSpec.Mode.PRETRAIN;
            case 1 -> RunSpec.Mode.FINETUNE;
            default -> throw new IOException("unknown run mode " + code);
        };
    }

    private static RunSpec readRun(DataInputStream in) throws IOException {
        return new RunSpec(
                modeFromCode(in.readByte()),
                ModelIO.readString(in),
                ModelIO.readString(in),
                in.readInt(),
                in.readDouble(),
                in.readDouble(),
                in.readInt(),
                in.readInt(),
                in.readInt(),
                in.readInt(),
                in.readBoolean(),
                in.readInt(),
                in.readDouble(),
                in.readBoolean(),
                ModelIO.readString(in),
                in.readInt());
    }

    // a serialized java.util.Random is about 100 bytes
    private static final int MAX_RANDOM_BYTES = 4096;

    // java.util.Random keeps its seed private; serialization is the supported way to save and restore its position
    private static void writeRandom(DataOutputStream out, Random random) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream objects = new ObjectOutputStream(bytes)) {
            objects.writeObject(random);
        }
        out.writeInt(bytes.size());
        bytes.writeTo(out);
    }

    // the read is safe: the filter below admits java.util.Random and nothing else, so no other class (and no gadget
    // chain) can be instantiated from the file, and the length check stops a corrupt length from allocating gigabytes
    @SuppressWarnings("BanSerializableRead")
    private static Random readRandom(DataInputStream in) throws IOException {
        int length = in.readInt();
        if (length < 0 || length > MAX_RANDOM_BYTES) {
            throw new IOException("corrupt checkpoint: random generator length " + length);
        }
        byte[] bytes = new byte[length];
        in.readFully(bytes);
        try (ObjectInputStream objects = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
            // deserializing a file can create any class on the classpath; allow only the one expected here
            objects.setObjectInputFilter(ObjectInputFilter.Config.createFilter("java.util.Random;!*"));
            return (Random) objects.readObject();
        } catch (ClassNotFoundException | ClassCastException e) {
            throw new IOException("stored random generator is not a java.util.Random", e);
        }
    }
}
