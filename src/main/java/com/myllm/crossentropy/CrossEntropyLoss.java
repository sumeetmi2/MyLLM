package com.myllm.crossentropy;

import com.myllm.dao.Matrix;
import java.util.stream.IntStream;

/**
 * Softmax cross-entropy between the logits and the next-token targets, averaged over the positions the mask
 * keeps.
 */
public class CrossEntropyLoss {

    private int[] targets;
    private Matrix logitsSoftMaxed;
    // null = every position counts
    private boolean[] mask;
    private int count;

    public double forwardPass(Matrix logits, int[] targets) {
        return forwardPass(logits, targets, null);
    }

    // mask[t] = false leaves position t out of the loss (e.g. prompt tokens during fine-tuning); null = all positions.
    // the loss is the average over the counted positions only
    public double forwardPass(Matrix logits, int[] targets, boolean[] mask) {
        this.count = countedPositions(logits, targets, mask);
        this.targets = targets;
        this.mask = mask;
        // per row: loss = log(sum exp(logit)) - logit[target], computed as m + log(sum exp(logit - m)).
        // taking -log of the stored float probability would give log(0) = -inf once a probability rounds to 0
        int T = logits.rows();
        int V = logits.cols();
        float[] z = logits.data();
        logitsSoftMaxed = new Matrix(T, V);
        float[] p = logitsSoftMaxed.data();
        double[] rowLoss = new double[T];
        IntStream.range(0, T).parallel().forEach(i -> {
            int start = i * V;
            float max = z[start];
            for (int j = 1; j < V; j++) {
                max = Math.max(max, z[start + j]);
            }
            double sumExp = 0.0;
            for (int j = 0; j < V; j++) {
                double e = Math.exp(z[start + j] - max);
                p[start + j] = (float) e;
                sumExp += e;
            }
            for (int j = 0; j < V; j++) {
                p[start + j] = (float) (p[start + j] / sumExp);
            }
            rowLoss[i] = max + Math.log(sumExp) - z[start + targets[i]];
        });
        // rows added in order so the total does not depend on thread scheduling
        double lossSum = 0.0;
        for (int i = 0; i < targets.length; i++) {
            if (mask == null || mask[i]) {
                lossSum += rowLoss[i];
            }
        }
        double loss = lossSum / count;
        return loss;
    }

    // checks the shapes and returns how many positions the mask keeps
    private static int countedPositions(Matrix logits, int[] targets, boolean[] mask) {
        if (targets.length != logits.rows()) {
            throw new IllegalArgumentException("targets length " + targets.length + " != logits rows " + logits.rows());
        }
        if (mask != null && mask.length != targets.length) {
            throw new IllegalArgumentException("mask length " + mask.length + " != targets length " + targets.length);
        }
        int counted = 0;
        for (int i = 0; i < targets.length; i++) {
            if (mask == null || mask[i]) {
                counted++;
            }
        }
        // 0 / 0 would be NaN and spread into every weight
        if (counted == 0) {
            throw new IllegalArgumentException("mask leaves no position to grade");
        }
        return counted;
    }

    // rows left out by the mask get zero gradient; the rest are (p - onehot) / count
    public Matrix backwardPass() {
        int T = logitsSoftMaxed.rows();
        int V = logitsSoftMaxed.cols();
        Matrix dLogits = new Matrix(T, V);
        float[] p = logitsSoftMaxed.data();
        float[] d = dLogits.data();
        IntStream.range(0, T).parallel().forEach(i -> {
            if (mask != null && !mask[i]) {
                return;
            }
            int start = i * V;
            for (int j = 0; j < V; j++) {
                float gradient = p[start + j];
                if (j == targets[i]) {
                    gradient -= 1.0f;
                }
                d[start + j] = gradient / count;
            }
        });
        return dLogits;
    }
}
