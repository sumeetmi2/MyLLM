package com.myllm.train;

import com.myllm.dao.Matrix;
import java.util.function.DoubleSupplier;

/**
 * Numerical gradient check: perturbs each element of a parameter and compares the finite difference to the
 * analytic gradient.
 */
public class GradCheck {

    // weights are float (~7 digits), so a 1e-5 step would drown in rounding; 1e-2 keeps the
    // central-difference error near H^2 = 1e-4 while staying well above float noise
    private static final double H = 1e-2;
    private static final double ABSOLUTE_NOISE_FLOOR = 1e-4;

    public record GradCheckResult(
            String name, double maxRelativeError, int row, int col, double analytic, double numerical) {
        public boolean passed(double tolerance) {
            return maxRelativeError < tolerance;
        }

        @Override
        public String toString() {
            return name + "[" + row + "][" + col + "] analytic=" + analytic + " numerical=" + numerical + " relErr="
                    + maxRelativeError;
        }
    }

    // loss must re-run forward pass so perturbed param takes effect. param is mutated in place and restored.
    public static GradCheckResult check(String name, Matrix param, Matrix analytic, DoubleSupplier loss) {
        GradCheckResult worst = new GradCheckResult(name, 0.0, -1, -1, 0.0, 0.0);
        float[] p = param.data();
        for (int idx = 0; idx < p.length; idx++) {
            float original = p[idx];
            float plus = (float) (original + H);
            float minus = (float) (original - H);
            p[idx] = plus;
            double lossPlus = loss.getAsDouble();
            p[idx] = minus;
            double lossMinus = loss.getAsDouble();
            p[idx] = original;

            // divide by the step actually taken: original +- H rounded to float is not exactly 2H apart
            double numerical = (lossPlus - lossMinus) / ((double) plus - minus);
            double a = analytic.data()[idx];
            double absoluteError = Math.abs(a - numerical);
            // float forward passes carry ~1e-5 rounding noise in the difference, so tiny gradients can show a large
            // relative error even when correct. Differences below that noise floor count as a match.
            double relativeError = absoluteError < ABSOLUTE_NOISE_FLOOR
                    ? 0.0
                    : absoluteError / Math.max(1e-8, Math.abs(a) + Math.abs(numerical));
            if (relativeError >= worst.maxRelativeError()) {
                worst = new GradCheckResult(name, relativeError, idx / param.cols(), idx % param.cols(), a, numerical);
            }
        }
        return worst;
    }

    // L = sum(y ⊙ dY), so dL/dy = dY exactly. Use as scalar loss when testing a single layer.
    public static double weightedSum(Matrix y, Matrix dY) {
        double sum = 0;
        float[] a = y.data();
        float[] b = dY.data();
        for (int k = 0; k < a.length; k++) {
            sum += (double) a[k] * b[k];
        }
        return sum;
    }
}
