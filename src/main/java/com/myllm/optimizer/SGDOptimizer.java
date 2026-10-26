package com.myllm.optimizer;

import com.myllm.dao.Matrix;
import java.util.List;

/**
 * Plain stochastic gradient descent: {@code w -= learningRate * gradient}.
 */
public class SGDOptimizer implements Optimizer {

    private double learningRate;

    public SGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    private void step(Matrix weights, Matrix gradients) {
        float[] w = weights.data();
        float[] g = gradients.data();
        for (int k = 0; k < w.length; k++) {
            w[k] -= (float) (g[k] * learningRate);
        }
    }

    @Override
    public void step(List<Matrix> parameters, List<Matrix> grads) {
        for (int i = 0; i < parameters.size(); i++) {
            step(parameters.get(i), grads.get(i));
        }
    }

    @Override
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
