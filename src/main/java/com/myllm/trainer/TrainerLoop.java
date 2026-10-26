package com.myllm.trainer;

import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.optimizer.Optimizer;
import com.myllm.optimizer.WarmupCosineScheduler;
import com.myllm.transformer.TransformerModel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * The simple one-sequence training loop: forward, loss, backward and optimizer step per sequence.
 *
 * <p>Pretraining draws a random {@code T}-token window from the corpus each step; supervised fine-tuning trains on
 * one whole chat example per step, with the loss on the answer only.
 */
public class TrainerLoop {

    private final TransformerModel model;
    private final CrossEntropyLoss loss;
    private final Optimizer optimiser;
    // exactly one of corpusIds / examples is set: random T-token windows over a corpus, or one chat example per step
    private final int[] corpusIds;
    private final List<ChatExample> examples;
    private final int T;
    private final Random random;
    private int completedSteps = 0;
    // null = keep the optimizer's own fixed learning rate
    private final WarmupCosineScheduler scheduler;
    private double lastLearningRate = Double.NaN;

    public TrainerLoop(
            TransformerModel model, CrossEntropyLoss loss, Optimizer optimiser, int[] corpusIds, int T, Random random) {
        this(model, loss, optimiser, corpusIds, T, random, null);
    }

    // T = context length per step (usually T_max). random should be seeded so training windows are reproducible
    public TrainerLoop(
            TransformerModel model,
            CrossEntropyLoss loss,
            Optimizer optimiser,
            int[] corpusIds,
            int T,
            Random random,
            WarmupCosineScheduler scheduler) {
        if (T <= 0) {
            throw new IllegalArgumentException("T must be positive, got " + T);
        }
        if (corpusIds.length < T + 1) {
            throw new IllegalArgumentException(
                    "corpus has " + corpusIds.length + " tokens, need at least T + 1 = " + (T + 1));
        }
        this.model = model;
        this.loss = loss;
        this.optimiser = optimiser;
        this.corpusIds = corpusIds;
        this.examples = null;
        this.T = T;
        this.random = random;
        this.scheduler = scheduler;
    }

    // supervised fine-tuning: every step is one whole chat example, graded only on its answer tokens
    public TrainerLoop(
            TransformerModel model,
            CrossEntropyLoss loss,
            Optimizer optimiser,
            List<ChatExample> examples,
            Random random,
            WarmupCosineScheduler scheduler) {
        if (examples.isEmpty()) {
            throw new IllegalArgumentException("no chat examples to train on");
        }
        this.model = model;
        this.loss = loss;
        this.optimiser = optimiser;
        this.corpusIds = null;
        this.examples = examples;
        this.T = 0;
        this.random = random;
        this.scheduler = scheduler;
    }

    // mask = which target positions count in the loss; null = all of them
    record Window(int[] input, int[] targets, boolean[] mask) {
        Window(int[] input, int[] targets) {
            this(input, targets, null);
        }
    }

    // targets = input shifted left by one: row t of the logits is graded on the token that follows position t
    Window window(int start) {
        int[] input = Arrays.copyOfRange(corpusIds, start, start + T);
        int[] targets = Arrays.copyOfRange(corpusIds, start + 1, start + T + 1);
        return new Window(input, targets);
    }

    // start in [0, len - T - 1] so the target slice (ending at start + T) stays inside the corpus
    Window randomWindow() {
        return window(random.nextInt(corpusIds.length - T));
    }

    // the whole example: input = ids[0 .. n-2], targets = ids[1 .. n-1].
    // targets[t] = ids[t+1] is an answer token when t + 1 >= answerStart;
    // the first counted position is the end of "Assistant:" predicting the first answer token
    static Window exampleWindow(ChatExample example) {
        int[] ids = example.ids();
        int[] input = Arrays.copyOfRange(ids, 0, ids.length - 1);
        int[] targets = Arrays.copyOfRange(ids, 1, ids.length);
        boolean[] mask = new boolean[targets.length];
        for (int t = 0; t < mask.length; t++) {
            mask[t] = t + 1 >= example.answerStart();
        }
        return new Window(input, targets, mask);
    }

    private Window nextWindow() {
        return examples != null ? exampleWindow(examples.get(random.nextInt(examples.size()))) : randomWindow();
    }

    // forward -> loss -> loss backward -> model backward -> update. No other forwardPass may run in between,
    // because every backward reads the caches written by the forward just before it.
    // step = 0-based index over the whole run (completedSteps), not the per-call loop counter,
    // otherwise the schedule restarts warmup every time train() is called
    public double trainStep() {
        Window w = nextWindow();
        double stepLoss = loss.forwardPass(model.forwardPass(w.input()), w.targets(), w.mask());
        TransformerModel.TransformerModelBackwardPassResult result = model.backwardPass(loss.backwardPass());
        if (scheduler != null) {
            lastLearningRate = scheduler.lr(completedSteps);
            optimiser.setLearningRate(lastLearningRate);
        }
        optimiser.step(model.trainableParameters(), result.trainableGradients());
        return stepLoss;
    }

    public int completedSteps() {
        return completedSteps;
    }

    // resuming from a checkpoint: the schedule continues from here. The Random passed to the constructor must be the
    // restored one too, so the next windows are the ones the uninterrupted run would have drawn
    public void restoreProgress(int completedSteps) {
        if (completedSteps < 0) {
            throw new IllegalArgumentException("completedSteps must be >= 0, got " + completedSteps);
        }
        this.completedSteps = completedSteps;
    }

    // returns the average loss of each logEvery-step chunk;
    // single-step losses are noisy because every step sees a different window
    // can be called repeatedly (e.g. to sample between chunks); step numbers in the log keep counting across calls
    public List<Double> train(int steps, int logEvery) {
        List<Double> loggedLosses = new ArrayList<>();
        double runningLoss = 0.0;
        for (int step = 1; step <= steps; step++) {
            runningLoss += trainStep();
            completedSteps++;
            if (step % logEvery == 0) {
                double average = runningLoss / logEvery;
                if (scheduler != null) {
                    System.out.printf("step %d  loss %.4f  lr %.6f%n", completedSteps, average, lastLearningRate);
                } else {
                    System.out.printf("step %d  loss %.4f%n", completedSteps, average);
                }
                loggedLosses.add(average);
                runningLoss = 0.0;
            }
        }
        return loggedLosses;
    }
}
