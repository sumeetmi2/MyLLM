package com.myllm.batched;

import java.util.List;

/**
 * Several sequences stacked row-wise: sequence {@code s} owns rows {@code [start(s), start(s) + lengths[s])} of
 * every activation.
 *
 * <p>There is no padding, so chat examples of different lengths stack as they are. {@code mask == null} means
 * every target counts.
 */
public record Batch(int[] ids, int[] lengths, int[] targets, boolean[] mask) {

    public Batch {
        int total = 0;
        for (int length : lengths) {
            if (length <= 0) {
                throw new IllegalArgumentException("sequence length must be positive, got " + length);
            }
            total += length;
        }
        if (total != ids.length || targets.length != ids.length) {
            throw new IllegalArgumentException("lengths sum to " + total + " but there are " + ids.length + " ids and "
                    + targets.length + " targets");
        }
        if (mask != null && mask.length != ids.length) {
            throw new IllegalArgumentException("mask length " + mask.length + " != " + ids.length + " ids");
        }
    }

    // masks may hold nulls (all positions count); the batch mask is null only when every sequence's mask is
    public static Batch stack(List<int[]> inputs, List<int[]> targets, List<boolean[]> masks) {
        int total = inputs.stream().mapToInt(a -> a.length).sum();
        boolean anyMask = masks.stream().anyMatch(m -> m != null);
        int[] ids = new int[total];
        int[] allTargets = new int[total];
        boolean[] mask = anyMask ? new boolean[total] : null;
        int[] lengths = new int[inputs.size()];
        int row = 0;
        for (int s = 0; s < inputs.size(); s++) {
            int[] in = inputs.get(s);
            lengths[s] = in.length;
            System.arraycopy(in, 0, ids, row, in.length);
            System.arraycopy(targets.get(s), 0, allTargets, row, in.length);
            if (anyMask) {
                boolean[] m = masks.get(s);
                for (int t = 0; t < in.length; t++) {
                    mask[row + t] = m == null || m[t];
                }
            }
            row += in.length;
        }
        return new Batch(ids, lengths, allTargets, mask);
    }

    // targets that count in the loss
    public int countedTokens() {
        if (mask == null) {
            return ids.length;
        }
        int counted = 0;
        for (boolean m : mask) {
            if (m) {
                counted++;
            }
        }
        return counted;
    }
}
