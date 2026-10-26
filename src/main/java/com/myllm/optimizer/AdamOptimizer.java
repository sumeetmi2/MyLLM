package com.myllm.optimizer;

import com.myllm.dao.Matrix;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Adam: a per-weight step size from running averages of the gradient ({@code m}) and squared gradient
 * ({@code v}).
 *
 * <p>State ({@code m}, {@code v}, {@code t}) lives across steps, so use one instance per model.
 */
public class AdamOptimizer implements Optimizer {
    private double learningRate;
    private final double beta1;
    private final double beta2;
    private final double eps;

    // step counter for bias correction, shared by all parameters
    private int t = 0;
    // one m and one v value per weight, laid out like each parameter's data; created on the first step.
    // stored as float, but each update is computed in double so eps = 1e-8 and tiny squared gradients survive
    private List<float[]> m;
    private List<float[]> v;

    public AdamOptimizer(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public AdamOptimizer(double learningRate, double beta1, double beta2, double eps) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
    }

    // everything a resumed run needs to continue exactly: the step counter and both moment arrays.
    // empty lists = no step taken yet
    public record State(int t, List<float[]> m, List<float[]> v) {}

    // copies, so later steps don't change a state that is being saved
    public State exportState() {
        if (m == null) {
            return new State(t, List.of(), List.of());
        }
        return new State(t, copy(m), copy(v));
    }

    // call before the first step; the next step must get parameters of the same count and sizes
    public void importState(State state) {
        if (state.m().size() != state.v().size()) {
            throw new IllegalArgumentException("state has " + state.m().size() + " m arrays but "
                    + state.v().size() + " v arrays");
        }
        for (int p = 0; p < state.m().size(); p++) {
            if (state.m().get(p).length != state.v().get(p).length) {
                throw new IllegalArgumentException(
                        "parameter " + p + ": m has " + state.m().get(p).length + " values, v has "
                                + state.v().get(p).length);
            }
        }
        t = state.t();
        m = state.m().isEmpty() ? null : copy(state.m());
        v = state.v().isEmpty() ? null : copy(state.v());
    }

    private static List<float[]> copy(List<float[]> arrays) {
        List<float[]> result = new ArrayList<>(arrays.size());
        for (float[] array : arrays) {
            result.add(array.clone());
        }
        return result;
    }

    public double getBeta1() {
        return beta1;
    }

    public double getBeta2() {
        return beta2;
    }

    public double getEps() {
        return eps;
    }

    // only scales the update; m, v and t are untouched, so changing it between steps is safe
    @Override
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void step(List<Matrix> parameters, List<Matrix> gradients) {
        if (parameters.size() != gradients.size()) {
            throw new IllegalArgumentException(
                    parameters.size() + " parameters but " + gradients.size() + " gradients");
        }
        if (m == null) {
            m = zerosLike(parameters);
            v = zerosLike(parameters);
        } else if (m.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "optimizer was set up for " + m.size() + " parameters, got " + parameters.size());
        }
        for (int p = 0; p < parameters.size(); p++) {
            // only possible after importState with a state from a different model
            if (m.get(p).length != parameters.get(p).data().length) {
                throw new IllegalArgumentException("optimizer state for parameter " + p + " has " + m.get(p).length
                        + " values, parameter has " + parameters.get(p).data().length);
            }
        }
        // once per step, not once per matrix
        t++;
        // m and v start at 0, so early averages are too small; dividing by these undoes that
        double mCorrection = 1 - Math.pow(beta1, t);
        double vCorrection = 1 - Math.pow(beta2, t);
        // each parameter has its own m / v, so parameters can update in parallel
        IntStream.range(0, parameters.size())
                .parallel()
                .forEach(p -> step(parameters.get(p), gradients.get(p), m.get(p), v.get(p), mCorrection, vCorrection));
    }

    private void step(
            Matrix weights, Matrix gradients, float[] mp, float[] vp, double mCorrection, double vCorrection) {
        if (gradients.rows() != weights.rows() || gradients.cols() != weights.cols()) {
            throw new IllegalArgumentException("gradient " + gradients.rows() + "x" + gradients.cols()
                    + " does not match parameter " + weights.rows() + "x" + weights.cols());
        }
        float[] w = weights.data();
        float[] g = gradients.data();
        for (int k = 0; k < w.length; k++) {
            double grad = g[k];
            double mk = beta1 * mp[k] + (1 - beta1) * grad;
            double vk = beta2 * vp[k] + (1 - beta2) * grad * grad;
            mp[k] = (float) mk;
            vp[k] = (float) vk;
            double mHat = mk / mCorrection;
            double vHat = vk / vCorrection;
            w[k] -= (float) (learningRate * mHat / (Math.sqrt(vHat) + eps));
        }
    }

    private static List<float[]> zerosLike(List<Matrix> parameters) {
        List<float[]> result = new ArrayList<>(parameters.size());
        for (Matrix parameter : parameters) {
            result.add(new float[parameter.data().length]);
        }
        return result;
    }
}
