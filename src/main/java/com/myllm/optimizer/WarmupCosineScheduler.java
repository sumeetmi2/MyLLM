package com.myllm.optimizer;

import static java.lang.Math.PI;

/**
 * Linear warmup from about 0 to {@code maxLr} over {@code warmUpSteps}, then cosine decay to {@code minLr} at
 * {@code totalSteps}, and {@code minLr} after that.
 *
 * <p>{@code step} is 0-based and counts across the whole run.
 */
public class WarmupCosineScheduler {

    private final double maxLr;
    private final double minLr;
    private final int warmUpSteps;
    private final int totalSteps;

    public WarmupCosineScheduler(double maxLr, double minLr, int warmUpSteps, int totalSteps) {
        if (maxLr <= 0 || minLr < 0 || minLr > maxLr) {
            throw new IllegalArgumentException(
                    "need 0 <= minLr <= maxLr and maxLr > 0, got minLr=" + minLr + " maxLr=" + maxLr);
        }
        if (warmUpSteps < 0 || warmUpSteps >= totalSteps) {
            throw new IllegalArgumentException(
                    "need 0 <= warmUpSteps < totalSteps, got " + warmUpSteps + " and " + totalSteps);
        }
        this.maxLr = maxLr;
        this.minLr = minLr;
        this.warmUpSteps = warmUpSteps;
        this.totalSteps = totalSteps;
    }

    public double getMaxLr() {
        return maxLr;
    }

    public double getMinLr() {
        return minLr;
    }

    public int getWarmUpSteps() {
        return warmUpSteps;
    }

    public int getTotalSteps() {
        return totalSteps;
    }

    public double lr(int step) {
        if (step < warmUpSteps) {
            return maxLr * ((double) (step + 1)) / warmUpSteps;
        }
        if (step >= totalSteps) {
            return minLr;
        }
        double progress = (double) (step - warmUpSteps) / (double) (totalSteps - warmUpSteps);
        return minLr + 0.5 * (maxLr - minLr) * (1 + Math.cos(PI * progress));
    }
}
