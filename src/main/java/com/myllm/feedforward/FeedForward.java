package com.myllm.feedforward;

import com.myllm.dao.Matrix;
import com.myllm.lora.LoraAdapter;
import java.util.Random;

/**
 * Position-wise feed-forward network: {@code ReLU(x W_1 + b_1) W_2 + b_2}, applied to every row independently.
 */
public class FeedForward {

    private Matrix W_1;
    private Matrix W_2;
    private Matrix B_1;
    private Matrix B_2;
    private int d_Model;
    private int d_ff;
    private Matrix activation;
    private Matrix cachedX;
    // null = LoRA off; loraW1 reads the FFN input, loraW2 reads the ReLU output
    private LoraAdapter loraW1;
    private LoraAdapter loraW2;

    public FeedForward(int d_Model, int d_ff, Random rnd, double initStdDev, double residualStdDev) {
        this.d_Model = d_Model;
        this.d_ff = d_ff;

        this.W_1 = new Matrix(d_Model, d_ff).normalFill(rnd, initStdDev);
        this.B_1 = new Matrix(1, d_ff).fillZeroes();
        this.W_2 = new Matrix(d_ff, d_Model).normalFill(rnd, residualStdDev);
        this.B_2 = new Matrix(1, d_Model).fillZeroes();
    }

    public void enableLora(int r, double alpha, Random rnd) {
        loraW1 = new LoraAdapter(d_Model, d_ff, r, alpha, rnd);
        loraW2 = new LoraAdapter(d_ff, d_Model, r, alpha, rnd);
    }

    // folds each adapter into its own weight and drops it
    public void mergeLora() {
        if (loraW1 != null) {
            loraW1.merge(W_1);
            loraW1 = null;
        }
        if (loraW2 != null) {
            loraW2.merge(W_2);
            loraW2 = null;
        }
    }

    public LoraAdapter getLoraW1() {
        return loraW1;
    }

    public LoraAdapter getLoraW2() {
        return loraW2;
    }

    public Matrix forwardPass(Matrix input) {
        this.cachedX = input;
        Matrix H = input.multiply(W_1);
        if (loraW1 != null) {
            H = H.add(loraW1.forwardPass(input));
        }
        H.addRowVectorInPlace(B_1);
        activation = H.relu();
        Matrix Y = activation.multiply(W_2);
        if (loraW2 != null) {
            Y = Y.add(loraW2.forwardPass(activation));
        }
        Y.addRowVectorInPlace(B_2);
        return Y;
    }

    public record FeedForwardBackwardPassResult(
            Matrix d_X,
            Matrix d_W_1,
            Matrix d_B_1,
            Matrix d_W_2,
            Matrix d_B_2,
            LoraAdapter.LoraAdapterBackwardPassResult loraW1Result,
            LoraAdapter.LoraAdapterBackwardPassResult loraW2Result) {}

    public FeedForwardBackwardPassResult backwardPass(Matrix dY) {
        Matrix dActivation = dY.multiplyTranspose(W_2);
        // loraW2 also read the activation, so its share of dActivation must be in before the ReLU mask below
        LoraAdapter.LoraAdapterBackwardPassResult loraW2Result = null;
        if (loraW2 != null) {
            loraW2Result = loraW2.backwardPass(activation, dY);
            dActivation = dActivation.add(loraW2Result.dX());
        }
        Matrix d_W_2 = activation.transposeMultiply(dY);
        Matrix d_B_2 = dY.columnSums();
        Matrix d_H = dActivation.elementWiseOp(activation.reluAndMask(), (g, m) -> g * m);
        Matrix d_W_1 = cachedX.transposeMultiply(d_H);
        Matrix d_X = d_H.multiplyTranspose(W_1);
        LoraAdapter.LoraAdapterBackwardPassResult loraW1Result = null;
        if (loraW1 != null) {
            loraW1Result = loraW1.backwardPass(cachedX, d_H);
            d_X = d_X.add(loraW1Result.dX());
        }
        Matrix d_B_1 = d_H.columnSums();
        return new FeedForwardBackwardPassResult(d_X, d_W_1, d_B_1, d_W_2, d_B_2, loraW1Result, loraW2Result);
    }

    public Matrix getW_1() {
        return W_1;
    }

    public Matrix getB_1() {
        return B_1;
    }

    public Matrix getW_2() {
        return W_2;
    }

    public Matrix getB_2() {
        return B_2;
    }
}
