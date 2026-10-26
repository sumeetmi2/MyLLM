package com.myllm.layernorm;

import com.myllm.dao.Matrix;
import java.util.stream.IntStream;

/**
 * Layer normalisation per row: {@code gamma * (x - mean) / sqrt(variance + eps) + beta}.
 */
public class LayerNorm {

    private Matrix gamma;
    private Matrix beta;
    private static final double EPSILON = 1e-5;
    private Matrix xHat = null;
    // one value per row of the last forward input
    private double[] invStd = null;
    private Matrix xCentered;

    public LayerNorm(int d_Model) {
        this.gamma = new Matrix(1, d_Model).fillWithValue(1.0);
        this.beta = new Matrix(1, d_Model).fillZeroes();
    }

    public Matrix forwardPass(Matrix x) {
        int rows = x.rows();
        int cols = x.cols();
        float[] in = x.data();
        xHat = new Matrix(rows, cols);
        invStd = new double[rows];
        xCentered = new Matrix(rows, cols);
        float[] xh = xHat.data();
        float[] xc = xCentered.data();
        float[] g = gamma.data();
        float[] b = beta.data();
        Matrix y = new Matrix(rows, cols);
        float[] out = y.data();
        // rows are independent; each task writes only row i (and invStd[i]).
        // mean and variance add d_model terms and the variance subtracts nearly equal numbers,
        // so the stats stay in double
        IntStream.range(0, rows).parallel().forEach(i -> {
            int start = i * cols;
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += in[start + j];
            }
            double mean = sum / cols;
            double squaredSum = 0.0;
            for (int j = 0; j < cols; j++) {
                squaredSum += Math.pow(in[start + j] - mean, 2.0);
            }
            double variance = squaredSum / cols;
            double invStdDeviation = 1 / Math.sqrt(variance + EPSILON);
            invStd[i] = invStdDeviation;
            for (int j = 0; j < cols; j++) {
                xc[start + j] = (float) (in[start + j] - mean);
                xh[start + j] = (float) (xc[start + j] * invStdDeviation);
                out[start + j] = g[j] * xh[start + j] + b[j];
            }
        });
        return y;
    }

    public record LayerNormBackwardResult(Matrix dX, Matrix dGamma, Matrix dBeta) {}

    public LayerNormBackwardResult backwardPass(Matrix dY) {
        int rows = dY.rows();
        int cols = dY.cols();
        float[] dy = dY.data();
        float[] xc = xCentered.data();
        float[] g = gamma.data();
        // dGamma / dBeta sum over rows, so they are column sums (computed per column, no shared writes)
        Matrix dGamma = dY.elementWiseOp(xHat, (d, xh) -> d * xh).columnSums();
        Matrix dBeta = dY.columnSums();
        Matrix dX = new Matrix(rows, cols);
        float[] dx = dX.data();
        // everything else is per row: each task reads row i and writes only dX row i.
        // the row scratch and sums stay in double, only the final dX is rounded to float
        IntStream.range(0, rows).parallel().forEach(i -> {
            int start = i * cols;
            double[] dXCenteredDirect = new double[cols];
            double dInvStd = 0.0;
            for (int j = 0; j < cols; j++) {
                double dXHat = dy[start + j] * g[j];
                dXCenteredDirect[j] = dXHat * invStd[i];
                dInvStd += dXHat * xc[start + j];
            }
            double dVariance = dInvStd * -0.5 * Math.pow(invStd[i], 3);
            double[] dXCenteredVariance = new double[cols];
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                dXCenteredVariance[j] = dVariance * (2.0 / cols) * xc[start + j];
                sum += dXCenteredVariance[j] + dXCenteredDirect[j];
            }
            for (int j = 0; j < cols; j++) {
                dx[start + j] = (float) (dXCenteredDirect[j] + dXCenteredVariance[j] - sum / cols);
            }
        });
        return new LayerNormBackwardResult(dX, dGamma, dBeta);
    }

    public Matrix getGamma() {
        return gamma;
    }

    public Matrix getBeta() {
        return beta;
    }
}
