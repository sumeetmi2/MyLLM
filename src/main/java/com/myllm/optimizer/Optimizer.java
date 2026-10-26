package com.myllm.optimizer;

import com.myllm.dao.Matrix;
import java.util.List;

/**
 * Updates parameters from their gradients; parameters and gradients are paired by list position.
 */
public interface Optimizer {
    void step(List<Matrix> parameters, List<Matrix> gradients);

    void setLearningRate(double learningRate);
}
