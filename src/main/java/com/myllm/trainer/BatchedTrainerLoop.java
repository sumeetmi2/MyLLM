package com.myllm.trainer;

import com.myllm.batched.Batch;
import com.myllm.batched.BatchedModel;
import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.dao.Matrix;
import com.myllm.optimizer.Optimizer;
import com.myllm.optimizer.WarmupCosineScheduler;
import com.myllm.transformer.TransformerModel;
import com.myllm.utils.Utils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * {@code TrainerLoop} with batches: each optimizer step runs {@code accumSteps} micro-batches of
 * {@code batchSize} sequences and adds up their gradients.
 *
 * <p>One step sees {@code batchSize * accumSteps} sequences while only one micro-batch of activations is held at a
 * time. The step's gradient is that of the mean loss over every counted token in the step. {@code TrainerLoop}
 * stays the simple one-sequence version.
 */
public class BatchedTrainerLoop {

    private final BatchedModel model;
    private final CrossEntropyLoss loss;
    private final Optimizer optimiser;
    // exactly one of corpusIds / examples is set, as in TrainerLoop
    private final int[] corpusIds;
    private final List<ChatExample> examples;
    private final int T;
    private final int batchSize;
    private final int accumSteps;
    private final Random random;
    private final WarmupCosineScheduler scheduler;
    private int completedSteps = 0;
    private long tokensSeen = 0;
    private double lastLearningRate = Double.NaN;

    // pretraining: random T-token windows over the corpus
    public BatchedTrainerLoop(
            TransformerModel model,
            CrossEntropyLoss loss,
            Optimizer optimiser,
            int[] corpusIds,
            int T,
            int batchSize,
            int accumSteps,
            Random random,
            WarmupCosineScheduler scheduler) {
        this(model, loss, optimiser, corpusIds, null, T, batchSize, accumSteps, random, scheduler);
        if (T <= 0) {
            throw new IllegalArgumentException("T must be positive, got " + T);
        }
        if (corpusIds.length < T + 1) {
            throw new IllegalArgumentException(
                    "corpus has " + corpusIds.length + " tokens, need at least T + 1 = " + (T + 1));
        }
    }

    // supervised fine-tuning: whole chat examples, graded only on their answer tokens
    public BatchedTrainerLoop(
            TransformerModel model,
            CrossEntropyLoss loss,
            Optimizer optimiser,
            List<ChatExample> examples,
            int batchSize,
            int accumSteps,
            Random random,
            WarmupCosineScheduler scheduler) {
        this(model, loss, optimiser, null, examples, 0, batchSize, accumSteps, random, scheduler);
        if (examples.isEmpty()) {
            throw new IllegalArgumentException("no chat examples to train on");
        }
    }

    private BatchedTrainerLoop(
            TransformerModel model,
            CrossEntropyLoss loss,
            Optimizer optimiser,
            int[] corpusIds,
            List<ChatExample> examples,
            int T,
            int batchSize,
            int accumSteps,
            Random random,
            WarmupCosineScheduler scheduler) {
        if (batchSize < 1 || accumSteps < 1) {
            throw new IllegalArgumentException(
                    "batchSize and accumSteps must be at least 1, got " + batchSize + " and " + accumSteps);
        }
        this.model = new BatchedModel(model);
        this.loss = loss;
        this.optimiser = optimiser;
        this.corpusIds = corpusIds;
        this.examples = examples;
        this.T = T;
        this.batchSize = batchSize;
        this.accumSteps = accumSteps;
        this.random = random;
        this.scheduler = scheduler;
    }

    // same draws as TrainerLoop, so batchSize = accumSteps = 1 trains on the same windows
    private TrainerLoop.Window nextWindow() {
        if (examples != null) {
            return TrainerLoop.exampleWindow(examples.get(random.nextInt(examples.size())));
        }
        int start = random.nextInt(corpusIds.length - T);
        return new TrainerLoop.Window(
                Arrays.copyOfRange(corpusIds, start, start + T),
                Arrays.copyOfRange(corpusIds, start + 1, start + T + 1));
    }

    Batch nextBatch() {
        List<int[]> inputs = new ArrayList<>(batchSize);
        List<int[]> targets = new ArrayList<>(batchSize);
        List<boolean[]> masks = new ArrayList<>(batchSize);
        for (int b = 0; b < batchSize; b++) {
            TrainerLoop.Window w = nextWindow();
            inputs.add(w.input());
            targets.add(w.targets());
            masks.add(w.mask());
        }
        return Batch.stack(inputs, targets, masks);
    }

    // summed gradients of one step: every micro-batch's loss gradient is scaled by its share of the counted tokens,
    // so the sum is the gradient of the mean over all of them, however the tokens are split into micro-batches
    record StepGradients(List<Matrix> gradients, double loss) {}

    StepGradients accumulate(List<Batch> microBatches) {
        long total = 0;
        for (Batch b : microBatches) {
            total += b.countedTokens();
        }
        List<Matrix> summed = null;
        double lossSum = 0.0;
        for (Batch b : microBatches) {
            int counted = b.countedTokens();
            lossSum += loss.forwardPass(model.forwardPass(b), b.targets(), b.mask()) * counted;
            Matrix dLogits = loss.backwardPass();
            if (counted != total) {
                dLogits = dLogits.scale((double) counted / total);
            }
            // each backward allocates fresh gradients, so the first micro-batch's can hold the running sum
            List<Matrix> gradients = model.backwardPass(dLogits).trainableGradients();
            if (summed == null) {
                summed = gradients;
            } else {
                for (int p = 0; p < summed.size(); p++) {
                    Utils.inPlaceAddition(summed.get(p), gradients.get(p));
                }
            }
            tokensSeen += b.ids().length;
        }
        return new StepGradients(summed, lossSum / total);
    }

    // samples every micro-batch first (SFT needs the step's total token count before the first backward),
    // then accumulates and takes one optimizer step. Returns the step's mean loss per counted token
    public double trainStep() {
        List<Batch> microBatches = new ArrayList<>(accumSteps);
        for (int m = 0; m < accumSteps; m++) {
            microBatches.add(nextBatch());
        }
        StepGradients step = accumulate(microBatches);
        if (scheduler != null) {
            lastLearningRate = scheduler.lr(completedSteps);
            optimiser.setLearningRate(lastLearningRate);
        }
        optimiser.step(model.getModel().trainableParameters(), step.gradients());
        return step.loss();
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

    // same logging as TrainerLoop.train; step numbers keep counting across calls
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

    // input tokens run through the model so far, for tokens/s
    public long tokensSeen() {
        return tokensSeen;
    }
}
