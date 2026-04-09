package tech.kayys.gollek.ml.nn.metrics;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Regression evaluation metrics: MAE, RMSE, R², MAPE.
 *
 * <p>All metrics operate on flat float arrays and use {@link VectorOps}
 * for SIMD-accelerated summation.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * float[] pred   = {1.0f, 2.0f, 3.0f};
 * float[] actual = {1.1f, 1.9f, 3.2f};
 * RegressionMetrics.Result r = RegressionMetrics.compute(pred, actual);
 * System.out.printf("MAE=%.4f RMSE=%.4f R²=%.4f%n", r.mae(), r.rmse(), r.r2());
 * }</pre>
 */
public final class RegressionMetrics {

    private RegressionMetrics() {}

    /**
     * Immutable result of a regression evaluation.
     *
     * @param mae  Mean Absolute Error
     * @param rmse Root Mean Squared Error
     * @param r2   Coefficient of Determination (R²)
     * @param mape Mean Absolute Percentage Error (in %)
     */
    public record Result(float mae, float rmse, float r2, float mape) {}

    /**
     * Computes all regression metrics in a single pass.
     *
     * @param predicted predicted values (length N)
     * @param actual    ground-truth values (length N)
     * @return {@link Result} with MAE, RMSE, R², MAPE
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static Result compute(float[] predicted, float[] actual) {
        if (predicted.length != actual.length)
            throw new IllegalArgumentException("Arrays must have the same length");
        int N = predicted.length;

        float[] absErr = new float[N], sqErr = new float[N], pctErr = new float[N];
        float meanActual = VectorOps.sum(actual) / N;
        float[] ssTot = new float[N];

        for (int i = 0; i < N; i++) {
            float diff = predicted[i] - actual[i];
            absErr[i] = Math.abs(diff);
            sqErr[i]  = diff * diff;
            pctErr[i] = actual[i] != 0 ? Math.abs(diff / actual[i]) * 100f : 0f;
            float dev = actual[i] - meanActual;
            ssTot[i]  = dev * dev;
        }

        float mae  = VectorOps.sum(absErr) / N;
        float mse  = VectorOps.sum(sqErr)  / N;
        float rmse = (float) Math.sqrt(mse);
        float ssRes = VectorOps.sum(sqErr);
        float ssTotSum = VectorOps.sum(ssTot);
        float r2   = ssTotSum > 0 ? 1f - ssRes / ssTotSum : 1f;
        float mape = VectorOps.sum(pctErr) / N;

        return new Result(mae, rmse, r2, mape);
    }

    /**
     * Computes regression metrics from {@link GradTensor} inputs.
     *
     * @param predicted predicted tensor (flat or any shape)
     * @param actual    ground-truth tensor (same shape)
     * @return {@link Result}
     */
    public static Result compute(GradTensor predicted, GradTensor actual) {
        return compute(predicted.data(), actual.data());
    }

    /**
     * Mean Absolute Error: {@code mean(|pred - actual|)}.
     *
     * @param predicted predicted values
     * @param actual    ground-truth values
     * @return MAE scalar
     */
    public static float mae(float[] predicted, float[] actual) {
        float[] abs = new float[predicted.length];
        for (int i = 0; i < predicted.length; i++) abs[i] = Math.abs(predicted[i] - actual[i]);
        return VectorOps.sum(abs) / predicted.length;
    }

    /**
     * Root Mean Squared Error: {@code sqrt(mean((pred - actual)²))}.
     *
     * @param predicted predicted values
     * @param actual    ground-truth values
     * @return RMSE scalar
     */
    public static float rmse(float[] predicted, float[] actual) {
        float[] sq = new float[predicted.length];
        for (int i = 0; i < predicted.length; i++) { float d = predicted[i]-actual[i]; sq[i] = d*d; }
        return (float) Math.sqrt(VectorOps.sum(sq) / predicted.length);
    }

    /**
     * Coefficient of Determination R²: {@code 1 - SS_res / SS_tot}.
     *
     * @param predicted predicted values
     * @param actual    ground-truth values
     * @return R² in (-∞, 1]; 1.0 = perfect prediction
     */
    public static float r2(float[] predicted, float[] actual) {
        return compute(predicted, actual).r2();
    }
}
