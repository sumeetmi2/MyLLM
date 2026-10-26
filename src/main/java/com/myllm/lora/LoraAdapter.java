package com.myllm.lora;

import com.myllm.dao.Matrix;
import com.myllm.utils.Utils;
import java.util.Random;

/**
 * Low-rank adapter for a frozen weight {@code W}: adds {@code (alpha / r) * x A B} to the layer output, where
 * {@code A} is {@code dIn x r} and {@code B} is {@code r x dOut}.
 *
 * <p>{@code B} starts at zero, so a new adapter leaves the model unchanged until it trains.
 */
public class LoraAdapter {
    // a and b are updated in place by the optimizer, so the references never change
    private final Matrix a;
    private final Matrix b;
    private final double scale;
    private Matrix xa;

    // b starts at zero, so a new adapter adds nothing and the model behaves exactly like the base
    // a is uniform +-1/sqrt(dIn): its scale follows the adapter's input width, not the base model's init std
    public LoraAdapter(int dIn, int dOut, int r, double alpha, Random rnd) {
        this.scale = alpha / r;
        this.a = new Matrix(dIn, r).uniformFill(rnd, 1.0 / Math.sqrt(dIn));
        this.b = new Matrix(r, dOut).fillZeroes();
    }

    public Matrix getA() {
        return this.a;
    }

    public Matrix getB() {
        return this.b;
    }

    public Matrix forwardPass(Matrix x) {
        this.xa = x.multiply(a);
        return xa.multiply(b).scale(scale);
    }

    public record LoraAdapterBackwardPassResult(Matrix dX, Matrix dA, Matrix dB) {}

    public LoraAdapterBackwardPassResult backwardPass(Matrix x, Matrix dY) {
        Matrix dB = xa.transposeMultiply(dY).scale(scale);
        Matrix dXA = dY.multiplyTranspose(b).scale(scale);
        Matrix dA = x.transposeMultiply(dXA);
        Matrix dX = dXA.multiplyTranspose(a);
        return new LoraAdapterBackwardPassResult(dX, dA, dB);
    }

    public void merge(Matrix w) {
        Matrix delta = a.multiply(b).scale(scale);
        Utils.inPlaceAddition(w, delta);
    }
}
