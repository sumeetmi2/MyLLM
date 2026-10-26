package com.myllm;

import com.myllm.corpus.CorpusCleaner;
import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.dao.Matrix;
import com.myllm.eval.Evaluator;
import com.myllm.generator.Generator;
import com.myllm.generator.InteractiveSession;
import com.myllm.optimizer.AdamOptimizer;
import com.myllm.optimizer.WarmupCosineScheduler;
import com.myllm.persist.CheckpointIO;
import com.myllm.persist.ModelIO;
import com.myllm.persist.RunSpec;
import com.myllm.persist.TrainingCheckpoint;
import com.myllm.server.ChatServer;
import com.myllm.tokenize.BpeTokenizer;
import com.myllm.tokenize.CharTokenizer;
import com.myllm.tokenize.Tokenizer;
import com.myllm.trainer.BatchedTrainerLoop;
import com.myllm.trainer.ChatExample;
import com.myllm.trainer.ChatExamples;
import com.myllm.trainer.TrainerLoop;
import com.myllm.transformer.TransformerModel;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.stream.Stream;

/**
 * End-to-end command line: train, fine-tune, resume, generate, chat, evaluate and serve models.
 *
 * <p>Run {@code ./gradlew run --args="help"} for every command and option; README.md carries the same text,
 * generated from {@link #main}'s usage output.
 */
public class Main {

    // hyperparameters (SPEC section 16)
    // T_MAX 256 BPE tokens fits a typical TinyStories-Instruct prompt + story
    // (~980 chars) for fine-tuning; D_FF = 4 * D_MODEL
    private static final int T_MAX = 256;
    private static final int D_MODEL = 128;
    private static final int HEADS = 4;
    private static final int D_FF = 512;
    private static final int BLOCKS = 4;
    private static final double LEARNING_RATE = 0.002;
    // cosine decay ends at a tenth of the peak; warmup is 5% of the run so it scales with the steps argument
    private static final double MIN_LEARNING_RATE = LEARNING_RATE / 10;
    private static final int WARMUP_DIVISOR = 20;
    private static final int STEPS = 2000;
    private static final int LOG_EVERY = 100;
    private static final int SAMPLE_EVERY = 500;
    private static final int SAMPLE_LENGTH = 200;
    private static final long SEED = 42;
    // chat sampling defaults: slightly sharpened distribution, only the 10 most likely chars
    private static final double CHAT_TEMPERATURE = 0.8;
    private static final int CHAT_TOP_K = 10;
    // BPE merges are learned from this much text; enough for stable merges, much faster than the full corpus
    private static final int BPE_SAMPLE_CHARS = 5_000_000;
    // fine-tuning starts from a trained model, so a full fine-tune uses a fifth of the pretraining peak;
    // LoRA needs bigger steps because B starts at zero and only ~2% of the weights train
    private static final int FINETUNE_STEPS = 3000;
    private static final double FINETUNE_LEARNING_RATE = 0.0004;
    private static final double LORA_LEARNING_RATE = 0.002;
    private static final String FINETUNE_SAMPLE_PROMPT = "User: Tell me a story about a cat.\nAssistant:";
    // eval: replies sampled per fixed prompt, with the chat sampling settings and a fixed seed so runs are comparable
    private static final int EVAL_SAMPLES = 10;
    private static final int EVAL_REPLY_LENGTH = 250;

    private static final String BUILT_IN_CORPUS = """
            the quick brown fox jumps over the lazy dog.
            a small model learns one character at a time.
            the fox runs and the dog sleeps in the sun.
            """.repeat(10);

    // batch / accum: sequences per micro-batch and micro-batches per optimizer step; 1 and 1 = the simple TrainerLoop.
    // checkpoint = file to save training state to every checkpointEvery steps (null = no checkpoints)
    private record RunOptions(int batch, int accum, Path checkpoint, int checkpointEvery) {
        static final RunOptions NONE = new RunOptions(1, 1, null, SAMPLE_EVERY);
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            train(BUILT_IN_CORPUS, null, STEPS, null, 0, RunOptions.NONE);
            return;
        }
        switch (args[0]) {
            case "train" -> {
                if (args.length < 3) {
                    usage();
                    return;
                }
                // optional args: [steps] [bpeVocab] in that order, "batch=B",
                // "accum=K", "ckpt=<file>", "ckptEvery=N" anywhere
                List<String> positional = new ArrayList<>();
                RunOptions options = parseOptions(args, 3, positional);
                int steps = positional.size() > 0 ? Integer.parseInt(positional.get(0)) : STEPS;
                int bpeVocab = positional.size() > 1 ? Integer.parseInt(positional.get(1)) : 0;
                train(pretrainText(Path.of(args[1])), args[1], steps, Path.of(args[2]), bpeVocab, options);
            }
            case "resume" -> {
                if (args.length < 2) {
                    usage();
                    return;
                }
                List<String> rest = new ArrayList<>();
                RunOptions options = parseOptions(args, 2, rest);
                if (!rest.isEmpty() || options.batch() != 1 || options.accum() != 1 || options.checkpoint() != null) {
                    // anything else would make the resumed run differ from the one that was interrupted
                    throw new IllegalArgumentException(
                            "resume takes only ckptEvery=N; the rest comes from the checkpoint");
                }
                boolean everyGiven = Arrays.stream(args).anyMatch(a -> a.startsWith("ckptEvery="));
                resume(Path.of(args[1]), everyGiven ? options.checkpointEvery() : 0);
            }
            case "generate" -> {
                if (args.length < 3) {
                    usage();
                    return;
                }
                int length = args.length > 3 ? Integer.parseInt(args[3]) : SAMPLE_LENGTH;
                ModelIO.LoadedModel loaded = ModelIO.load(Path.of(args[1]));
                System.out.println(new Generator(loaded.model(), loaded.tokenizer()).generate(args[2], length));
            }
            case "chat" -> {
                if (args.length < 2) {
                    usage();
                    return;
                }
                // optional args: [length] [temperature] [topK] in that order, "template" anywhere among them
                boolean template = false;
                List<String> positional = new ArrayList<>();
                for (int i = 2; i < args.length; i++) {
                    if (args[i].equals("template")) {
                        template = true;
                    } else {
                        positional.add(args[i]);
                    }
                }
                int length = positional.size() > 0 ? Integer.parseInt(positional.get(0)) : SAMPLE_LENGTH;
                double temperature = positional.size() > 1 ? Double.parseDouble(positional.get(1)) : CHAT_TEMPERATURE;
                int topK = positional.size() > 2 ? Integer.parseInt(positional.get(2)) : CHAT_TOP_K;
                ModelIO.LoadedModel loaded = ModelIO.load(Path.of(args[1]));
                Generator generator = new Generator(loaded.model(), loaded.tokenizer());
                System.out.printf("sampling: temperature=%.2f topK=%d%n", temperature, topK);
                new InteractiveSession(generator, loaded.tokenizer(), length, topK, temperature, template)
                        .run(new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)), System.out);
            }
            case "eval" -> {
                if (args.length < 3) {
                    usage();
                    return;
                }
                int samples = args.length > 3 ? Integer.parseInt(args[3]) : EVAL_SAMPLES;
                evaluate(Path.of(args[1]), Path.of(args[2]), samples);
            }
            case "finetune" -> {
                if (args.length < 4) {
                    usage();
                    return;
                }
                // optional args: [steps] [peakLr] in that order, "lora=r",
                // "lora-all", "sft", "batch=B" and "accum=K" anywhere among them
                int loraRank = 0;
                boolean loraAll = false;
                boolean sft = false;
                List<String> rest = new ArrayList<>();
                RunOptions options = parseOptions(args, 4, rest);
                List<String> positional = new ArrayList<>();
                for (String arg : rest) {
                    if (arg.equals("sft")) {
                        sft = true;
                    } else if (arg.equals("lora-all")) {
                        loraAll = true;
                    } else if (arg.startsWith("lora=")) {
                        loraRank = Integer.parseInt(arg.substring("lora=".length()));
                    } else {
                        positional.add(arg);
                    }
                }
                int steps = positional.size() > 0 ? Integer.parseInt(positional.get(0)) : FINETUNE_STEPS;
                double defaultLr = loraRank > 0 ? LORA_LEARNING_RATE : FINETUNE_LEARNING_RATE;
                double peakLr = positional.size() > 1 ? Double.parseDouble(positional.get(1)) : defaultLr;
                if (loraAll && loraRank == 0) {
                    throw new IllegalArgumentException("lora-all needs lora=r");
                }
                finetune(
                        Path.of(args[1]),
                        Path.of(args[2]),
                        Path.of(args[3]),
                        steps,
                        peakLr,
                        loraRank,
                        loraAll,
                        sft,
                        options);
            }
            case "serve" -> {
                if (args.length < 2) {
                    usage();
                    return;
                }
                int port = args.length > 2 ? Integer.parseInt(args[2]) : 8080;
                new ChatServer(Path.of(args[1])).start(port);
            }
            default -> usage();
        }
    }

    // pulls "batch=B", "accum=K", "ckpt=<file>", "ckptEvery=N" out of args[from..];
    // everything else goes to rest in order
    private static RunOptions parseOptions(String[] args, int from, List<String> rest) {
        int batch = 1;
        int accum = 1;
        Path checkpoint = null;
        int checkpointEvery = SAMPLE_EVERY;
        for (int i = from; i < args.length; i++) {
            if (args[i].startsWith("batch=")) {
                batch = Integer.parseInt(args[i].substring("batch=".length()));
            } else if (args[i].startsWith("accum=")) {
                accum = Integer.parseInt(args[i].substring("accum=".length()));
            } else if (args[i].startsWith("ckpt=")) {
                checkpoint = Path.of(args[i].substring("ckpt=".length()));
            } else if (args[i].startsWith("ckptEvery=")) {
                checkpointEvery = Integer.parseInt(args[i].substring("ckptEvery=".length()));
            } else {
                rest.add(args[i]);
            }
        }
        if (batch < 1 || accum < 1) {
            throw new IllegalArgumentException("batch and accum must be at least 1");
        }
        checkCheckpointEvery(checkpointEvery);
        return new RunOptions(batch, accum, checkpoint, checkpointEvery);
    }

    // checkpoints only where no log chunk is half done, so a resumed run logs exactly what the uninterrupted one did
    private static void checkCheckpointEvery(int checkpointEvery) {
        if (checkpointEvery <= 0 || checkpointEvery % LOG_EVERY != 0) {
            throw new IllegalArgumentException(
                    "ckptEvery must be a positive multiple of " + LOG_EVERY + ", got " + checkpointEvery);
        }
    }

    private static String pretrainText(Path corpusPath) throws IOException {
        return cleanCorpus(readCorpus(corpusPath));
    }

    // cleanDocument, not clean(): clean() is meant for markdown files and drops repeated paragraphs across documents
    private static String finetuneText(Path corpusPath) throws IOException {
        StringBuilder corpus = new StringBuilder();
        for (String document : readCorpus(corpusPath)) {
            corpus.append(CorpusCleaner.cleanDocument(document)).append("\n");
        }
        return corpus.toString();
    }

    // trains on text, prints loss + samples, and saves to savePath when given.
    // bpeVocab > 0 uses a BPE tokenizer with that many tokens, otherwise one token per char.
    // corpusPath = where text came from (null for the built-in demo text, which can't be checkpointed)
    private static void train(
            String text, String corpusPath, int steps, Path savePath, int bpeVocab, RunOptions options)
            throws IOException {
        Tokenizer tokenizer = buildTokenizer(text, bpeVocab);
        int vocabSize = tokenizer.vocabSize();
        int[] ids = tokenizer.encode(text);
        // loss is per token; dividing by chars per token makes BPE and char runs comparable
        double charsPerToken = (double) text.length() / ids.length;
        TransformerModel model = new TransformerModel(vocabSize, T_MAX, D_MODEL, HEADS, D_FF, BLOCKS, new Random(SEED));
        System.out.printf(
                "corpus: %d chars = %d tokens (%.2f chars/token), vocab V=%d, parameters=%d%n",
                text.length(), ids.length, charsPerToken, vocabSize, countParameters(model.parameters()));
        System.out.printf("expected loss at init ~ ln(V) = %.4f%n", Math.log(vocabSize));
        if (options.checkpoint() != null && corpusPath == null) {
            throw new IllegalArgumentException("the built-in demo text can't be checkpointed");
        }
        RunSpec run = new RunSpec(
                RunSpec.Mode.PRETRAIN,
                corpusPath == null ? "" : corpusPath,
                savePath == null ? "" : savePath.toString(),
                steps,
                LEARNING_RATE,
                MIN_LEARNING_RATE,
                Math.max(1, steps / WARMUP_DIVISOR),
                options.batch(),
                options.accum(),
                bpeVocab,
                false,
                0,
                0,
                false,
                text.substring(0, 1),
                options.checkpointEvery());
        runTraining(model, tokenizer, ids, null, charsPerToken, run, options.checkpoint(), null);
    }

    // continues training a saved model on new text with its own tokenizer and T_max; loraRank > 0 trains only
    // LoRA adapters (alpha = 2r) and merges them into the weights before saving, otherwise every weight trains.
    // sft: the corpus is chat examples, each step trains on one whole example and only its answer counts in the loss
    private static void finetune(
            Path basePath,
            Path corpusPath,
            Path savePath,
            int steps,
            double peakLr,
            int loraRank,
            boolean loraAll,
            boolean sft,
            RunOptions options)
            throws IOException {
        // overwriting the base would throw away the pretraining
        if (basePath.toAbsolutePath()
                .normalize()
                .equals(savePath.toAbsolutePath().normalize())) {
            throw new IllegalArgumentException("output " + savePath + " must differ from the base model " + basePath);
        }
        ModelIO.LoadedModel loaded = ModelIO.load(basePath);
        TransformerModel model = loaded.model();
        // never a new tokenizer: the embedding rows belong to the base tokenizer's ids
        Tokenizer tokenizer = loaded.tokenizer();
        TransformerModel.ModelConfig config = model.getConfig();
        System.out.printf(
                "base model %s: V=%d T_max=%d d_model=%d blocks=%d parameters=%d%n",
                basePath,
                config.vocabSize(),
                config.T_max(),
                config.d_Model(),
                config.noOfDecoderBlocks(),
                countParameters(model.parameters()));

        String text = finetuneText(corpusPath);
        int[] ids;
        try {
            ids = tokenizer.encode(text);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "corpus contains text the base tokenizer can't encode: " + e.getMessage(), e);
        }
        double charsPerToken = (double) text.length() / ids.length;
        System.out.printf(
                "corpus: %d chars = %d tokens (%.2f chars/token)%n", text.length(), ids.length, charsPerToken);
        List<ChatExample> examples = null;
        if (sft) {
            ChatExamples.Result built = ChatExamples.build(text, tokenizer, config.T_max());
            examples = built.examples();
            double avgTokens =
                    examples.stream().mapToInt(e -> e.ids().length).average().orElse(0);
            System.out.printf(
                    "chat examples: kept %d, too long %d, no Assistant marker %d, avg tokens %.0f (loss on answers only)%n",
                    examples.size(), built.tooLong(), built.noMarker(), avgTokens);
        }

        // before step 0, so the samples during training can be compared with what the base model did
        Generator baseline = new Generator(model, tokenizer, new Random(SEED));
        System.out.printf(
                "%n---prompt:%s base model before fine-tuning ---%n%s%n%n",
                FINETUNE_SAMPLE_PROMPT,
                baseline.generate(FINETUNE_SAMPLE_PROMPT, SAMPLE_LENGTH, CHAT_TOP_K, CHAT_TEMPERATURE));

        // enabled before runTraining creates the optimizer: Adam sizes its state from the first parameter list it sees
        if (loraRank > 0) {
            model.enableLora(loraRank, 2.0 * loraRank, loraAll, new Random(SEED));
            long trainable = countParameters(model.trainableParameters());
            System.out.printf(
                    "lora r=%d alpha=%d targets=%s trainable=%d/%d (%.1f%%)%n",
                    loraRank,
                    2 * loraRank,
                    loraAll ? "W_q,W_v,W_o,W_1,W_2" : "W_q,W_v",
                    trainable,
                    countParameters(model.parameters()),
                    100.0 * trainable / countParameters(model.parameters()));
        }
        RunSpec run = new RunSpec(
                RunSpec.Mode.FINETUNE,
                corpusPath.toString(),
                savePath.toString(),
                steps,
                peakLr,
                peakLr / 10,
                Math.max(1, steps / WARMUP_DIVISOR),
                options.batch(),
                options.accum(),
                0,
                sft,
                loraRank,
                2.0 * loraRank,
                loraAll,
                FINETUNE_SAMPLE_PROMPT,
                options.checkpointEvery());
        runTraining(model, tokenizer, ids, examples, charsPerToken, run, options.checkpoint(), null);
    }

    // continues an interrupted run from its checkpoint: same corpus (checked by fingerprint), same tokenizer,
    // weights, Adam state, schedule position and random generators, so it ends with the model the uninterrupted
    // run would have made. checkpointEvery 0 = keep the run's own setting
    private static void resume(Path checkpointPath, int checkpointEvery) throws IOException {
        TrainingCheckpoint checkpoint = CheckpointIO.load(checkpointPath);
        RunSpec run = checkpoint.run();
        if (checkpointEvery > 0) {
            run = new RunSpec(
                    run.mode(),
                    run.corpusPath(),
                    run.savePath(),
                    run.totalSteps(),
                    run.peakLr(),
                    run.minLr(),
                    run.warmupSteps(),
                    run.batch(),
                    run.accum(),
                    run.bpeVocab(),
                    run.sft(),
                    run.loraRank(),
                    run.loraAlpha(),
                    run.loraAll(),
                    run.samplePrompt(),
                    checkpointEvery);
        }
        Tokenizer tokenizer = checkpoint.tokenizer();
        System.out.printf(
                "resuming %s: %s run, step %d of %d, corpus %s%n",
                checkpointPath,
                run.mode().name().toLowerCase(Locale.ROOT),
                checkpoint.completedSteps(),
                run.totalSteps(),
                run.corpusPath());
        // the corpus is re-read and encoded with the stored tokenizer (BPE is never re-learned)
        String text = run.mode() == RunSpec.Mode.PRETRAIN
                ? pretrainText(Path.of(run.corpusPath()))
                : finetuneText(Path.of(run.corpusPath()));
        int[] ids = tokenizer.encode(text);
        if (ids.length != checkpoint.corpusTokens() || CheckpointIO.corpusCrc(ids) != checkpoint.corpusCrc()) {
            throw new IllegalStateException("corpus changed since checkpoint: " + ids.length + " tokens now, "
                    + checkpoint.corpusTokens() + " when the checkpoint was written");
        }
        double charsPerToken = (double) text.length() / ids.length;
        List<ChatExample> examples = run.sft()
                ? ChatExamples.build(
                                text, tokenizer, checkpoint.model().getConfig().T_max())
                        .examples()
                : null;
        runTraining(checkpoint.model(), tokenizer, ids, examples, charsPerToken, run, checkpointPath, checkpoint);
    }

    // masked validation loss on chat examples + plot-word hit rates on fixed prompts; same output format for
    // every model, so fine-tuning runs can be compared by numbers instead of single samples
    private static void evaluate(Path modelPath, Path corpusPath, int samples) throws IOException {
        ModelIO.LoadedModel loaded = ModelIO.load(modelPath);
        TransformerModel model = loaded.model();
        StringBuilder corpus = new StringBuilder();
        for (String document : readCorpus(corpusPath)) {
            corpus.append(CorpusCleaner.cleanDocument(document)).append("\n");
        }
        ChatExamples.Result built = ChatExamples.build(
                corpus.toString(), loaded.tokenizer(), model.getConfig().T_max());
        System.out.printf(
                "model %s, %d validation examples (%d too long, skipped)%n",
                modelPath, built.examples().size(), built.tooLong());

        long startNanos = System.nanoTime();
        Evaluator.LossResult loss = Evaluator.validationLoss(model, built.examples());
        System.out.printf(
                "validation loss (answer tokens only): %.4f over %d tokens (%.1fs)%n",
                loss.loss(), loss.answerTokens(), (System.nanoTime() - startNanos) / 1e9);

        System.out.printf(
                "%nplot prompts: %d samples each, temperature=%.2f topK=%d, seed %d%n",
                samples, CHAT_TEMPERATURE, CHAT_TOP_K, SEED);
        double score = Evaluator.plotReport(
                model, loaded.tokenizer(), samples, EVAL_REPLY_LENGTH, CHAT_TOP_K, CHAT_TEMPERATURE, SEED, System.out);
        System.out.printf("%nsummary %s: val_loss %.4f  plot_score %.3f%n", modelPath, loss.loss(), score);
    }

    // shared by train, finetune and resume: warmup + cosine schedule, logs loss,
    // prints a sample every SAMPLE_EVERY steps, saves to run.savePath() when set.
    // Shapes come from the model, not the constants, so older models keep their own T_max.
    // examples != null trains on chat examples (sft) instead of random windows over ids.
    // checkpointPath != null saves training state every run.checkpointEvery() steps and at the end;
    // resumed != null continues from that checkpoint's optimizer state, step and random generators
    // one linear script of the whole run (setup, chunked training loop, samples, checkpoints, save);
    // splitting it would scatter the run's state across helpers that only one caller uses
    @SuppressWarnings({"checkstyle:CyclomaticComplexity", "checkstyle:NPathComplexity", "checkstyle:MethodLength"})
    private static void runTraining(
            TransformerModel model,
            Tokenizer tokenizer,
            int[] ids,
            List<ChatExample> examples,
            double charsPerToken,
            RunSpec run,
            Path checkpointPath,
            TrainingCheckpoint resumed)
            throws IOException {
        TransformerModel.ModelConfig config = model.getConfig();
        int steps = run.totalSteps();
        checkCheckpointEvery(run.checkpointEvery());
        WarmupCosineScheduler scheduler =
                new WarmupCosineScheduler(run.peakLr(), run.minLr(), run.warmupSteps(), steps);
        AdamOptimizer optimizer = new AdamOptimizer(run.peakLr());
        Random trainerRandom = new Random(SEED);
        // samples use the same temperature + top-k as chat; greedy samples loop and hide real progress
        Random generatorRandom = new Random(SEED);
        int startStep = 0;
        if (resumed != null) {
            if (resumed.beta1() != optimizer.getBeta1()
                    || resumed.beta2() != optimizer.getBeta2()
                    || resumed.eps() != optimizer.getEps()) {
                throw new IllegalStateException("checkpoint was written with different Adam settings");
            }
            optimizer.importState(resumed.adam());
            trainerRandom = resumed.trainerRandom();
            generatorRandom = resumed.generatorRandom();
            startStep = resumed.completedSteps();
        }
        long corpusCrc = checkpointPath != null ? CheckpointIO.corpusCrc(ids) : 0;

        // batch=1 accum=1 keeps the simple one-sequence TrainerLoop;
        // anything else stacks sequences with BatchedTrainerLoop
        boolean batched = run.batch() > 1 || run.accum() > 1;
        TrainerLoop trainer = null;
        BatchedTrainerLoop batchedTrainer = null;
        if (!batched) {
            trainer = examples != null
                    ? new TrainerLoop(model, new CrossEntropyLoss(), optimizer, examples, trainerRandom, scheduler)
                    : new TrainerLoop(
                            model, new CrossEntropyLoss(), optimizer, ids, config.T_max(), trainerRandom, scheduler);
            trainer.restoreProgress(startStep);
        } else {
            batchedTrainer = examples != null
                    ? new BatchedTrainerLoop(
                            model,
                            new CrossEntropyLoss(),
                            optimizer,
                            examples,
                            run.batch(),
                            run.accum(),
                            trainerRandom,
                            scheduler)
                    : new BatchedTrainerLoop(
                            model,
                            new CrossEntropyLoss(),
                            optimizer,
                            ids,
                            config.T_max(),
                            run.batch(),
                            run.accum(),
                            trainerRandom,
                            scheduler);
            batchedTrainer.restoreProgress(startStep);
        }
        Generator generator = new Generator(model, tokenizer, generatorRandom);

        System.out.printf(
                "config: T_max=%d d_model=%d heads=%d d_ff=%d blocks=%d lr=%.4f->%.4f (warmup %d) steps=%d%n%n",
                config.T_max(),
                config.d_Model(),
                config.heads(),
                config.d_ff(),
                config.noOfDecoderBlocks(),
                run.peakLr(),
                run.minLr(),
                run.warmupSteps(),
                steps);
        if (batched) {
            System.out.printf(
                    "batching: batch=%d accum=%d (%d sequences per step)%n%n",
                    run.batch(), run.accum(), run.batch() * run.accum());
        }
        if (checkpointPath != null) {
            System.out.printf("checkpoints: %s every %d steps%n%n", checkpointPath, run.checkpointEvery());
        }
        if (resumed != null) {
            System.out.printf("resumed at step %d%n%n", startStep);
        }

        long startNanos = System.nanoTime();
        double lastLoss = Double.NaN;
        // the loss line only follows chunks that logged something, as before checkpoints split chunks
        boolean loggedSinceSample = false;
        for (int done = startStep; done < steps; ) {
            // stop at the next sample, checkpoint or end, whichever comes first
            int next = Math.min(steps, (done / SAMPLE_EVERY + 1) * SAMPLE_EVERY);
            if (checkpointPath != null) {
                next = Math.min(next, (done / run.checkpointEvery() + 1) * run.checkpointEvery());
            }
            int chunk = next - done;
            List<Double> losses =
                    trainer != null ? trainer.train(chunk, LOG_EVERY) : batchedTrainer.train(chunk, LOG_EVERY);
            if (!losses.isEmpty()) {
                lastLoss = losses.get(losses.size() - 1);
                loggedSinceSample = true;
            }
            done = next;
            if (done % SAMPLE_EVERY == 0 || done == steps) {
                if (loggedSinceSample) {
                    System.out.printf("loss %.4f per token = %.4f per char%n", lastLoss, lastLoss / charsPerToken);
                    loggedSinceSample = false;
                }
                System.out.printf(
                        "%n---prompt:%s sample after %d steps ---%n%s%n%n",
                        run.samplePrompt(),
                        done,
                        generator.generate(run.samplePrompt(), SAMPLE_LENGTH, CHAT_TOP_K, CHAT_TEMPERATURE));
            }
            // after the sample, so the stored generator state includes it
            if (checkpointPath != null && (done % run.checkpointEvery() == 0 || done == steps)) {
                long saveStart = System.nanoTime();
                CheckpointIO.save(
                        checkpointPath,
                        new TrainingCheckpoint(
                                run,
                                ids.length,
                                corpusCrc,
                                model,
                                tokenizer,
                                optimizer.exportState(),
                                optimizer.getBeta1(),
                                optimizer.getBeta2(),
                                optimizer.getEps(),
                                done,
                                trainerRandom,
                                generatorRandom));
                System.out.printf(
                        "checkpoint step %d -> %s (%.1fMB, %dms)%n%n",
                        done,
                        checkpointPath,
                        Files.size(checkpointPath) / 1e6,
                        (System.nanoTime() - saveStart) / 1_000_000);
            }
        }
        double seconds = (System.nanoTime() - startNanos) / 1e9;
        if (batchedTrainer != null) {
            System.out.printf(
                    "trained %d steps in %.1fs (%,.0f tokens/s)%n",
                    steps - startStep, seconds, batchedTrainer.tokensSeen() / seconds);
        } else {
            System.out.printf("trained %d steps in %.1fs%n", steps - startStep, seconds);
        }

        if (!run.savePath().isEmpty()) {
            Path savePath = Path.of(run.savePath());
            // ModelIO saves only the base weights, so adapters must be folded in first
            if (model.isLoraEnabled()) {
                model.mergeLora();
                System.out.println("merged LoRA adapters into base weights");
            }
            ModelIO.save(savePath, model, tokenizer);
            System.out.printf("saved model to %s (%d bytes)%n", savePath.toAbsolutePath(), Files.size(savePath));
        }
    }

    private static Tokenizer buildTokenizer(String text, int bpeVocab) {
        if (bpeVocab <= 0) {
            return new CharTokenizer(text);
        }
        long start = System.nanoTime();
        BpeTokenizer bpe = BpeTokenizer.train(text, bpeVocab, BPE_SAMPLE_CHARS);
        System.out.printf(
                "BPE tokenizer: %d base chars + %d merges = %d tokens (learned in %.1fs)%n",
                bpe.baseChars().length(), bpe.merges().size(), bpe.vocabSize(), (System.nanoTime() - start) / 1e9);
        return bpe;
    }

    // required before training on files: removes noise, then reports how much the text and vocab shrank
    private static String cleanCorpus(List<String> documents) {
        String raw = String.join("\n\n", documents);
        String cleaned = CorpusCleaner.clean(documents);
        if (cleaned.isEmpty()) {
            throw new IllegalArgumentException("corpus is empty after cleaning");
        }
        System.out.printf(
                "cleaned corpus: %d -> %d chars, vocab %d -> %d%n",
                raw.length(), cleaned.length(), distinctChars(raw), distinctChars(cleaned));
        return cleaned;
    }

    private static long distinctChars(String text) {
        return text.chars().distinct().count();
    }

    // a single file, or every .md file under a directory (sorted); one entry per file
    private static List<String> readCorpus(Path path) throws IOException {
        if (!Files.isDirectory(path)) {
            return List.of(Files.readString(path));
        }
        try (Stream<Path> files = Files.walk(path)) {
            List<Path> markdown = files.filter(Files::isRegularFile)
                    .filter(f -> f.getFileName().toString().endsWith(".md"))
                    .sorted()
                    .toList();
            if (markdown.isEmpty()) {
                throw new IllegalArgumentException("no .md files under " + path);
            }
            List<String> documents = new ArrayList<>();
            for (Path file : markdown) {
                documents.add(Files.readString(file));
            }
            System.out.printf("read %d .md files from %s%n", markdown.size(), path);
            return documents;
        }
    }

    private static void usage() {
        System.out.println("""
                usage:
                  (no args)                                   demo: train on built-in text, no save
                  train <corpus file|dir> <model.bin> [steps] [bpeVocab] [batch=B] [accum=K] [ckpt=<file>] [ckptEvery=N]
                                                              clean, train and save (dir = all .md files under it)
                                                              bpeVocab e.g. 1024 = BPE tokens, omitted = one token per char
                                                              batch=B sequences per micro-batch, accum=K micro-batches
                                                              per optimizer step (default 1 and 1)
                                                              ckpt=<file> saves training state every ckptEvery steps
                                                              (default 500) and at the end, for resume
                  resume <file.ckpt> [ckptEvery=N]            continue an interrupted train/finetune run exactly
                  serve <model.bin> [port]                    chat in the browser at http://localhost:8080 (default port);
                                                              the page can switch to any .bin next to the model
                  generate <model.bin> <prompt> [length]      load a saved model and generate text
                  chat <model.bin> [length] [temperature] [topK] [template]
                                                              interactive: type prompts in the terminal
                                                              (defaults 0.8, 10; temperature 0 = greedy)
                                                              template = send "User: <line>\\nAssistant:", print the reply only
                  eval <model.bin> <chat corpus> [samples]    validation loss on answers + plot-word hits on fixed prompts
                                                              (default 10 samples per prompt)
                  finetune <base.bin> <corpus> <out.bin> [steps] [peakLr] [lora=r] [lora-all] [sft]
                                                              continue training a saved model on new text (same tokenizer)
                                                              defaults 3000 steps, lr 0.0004 (0.002 with lora=r)
                                                              lora=r trains only rank-r adapters on W_q/W_v, merged before saving
                                                              lora-all also adapts W_o and the FFN (W_1, W_2)
                                                              sft = corpus is User:/Assistant: examples, one per step,
                                                              loss on the answer only
                                                              batch=B / accum=K / ckpt=<file> / ckptEvery=N as for train""");
    }

    private static long countParameters(List<Matrix> parameters) {
        long count = 0;
        for (Matrix parameter : parameters) {
            count += (long) parameter.rows() * parameter.cols();
        }
        return count;
    }
}
