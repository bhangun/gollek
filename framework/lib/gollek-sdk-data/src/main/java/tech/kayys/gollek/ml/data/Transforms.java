package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Composable image/tensor transform pipeline — equivalent to {@code torchvision.transforms.Compose}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var transform = Transforms.compose(
 *     Transforms.randomHorizontalFlip(0.5f),
 *     Transforms.normalize(new float[]{0.485f,0.456f,0.406f},
 *                          new float[]{0.229f,0.224f,0.225f})
 * );
 * GradTensor augmented = transform.apply(image);
 * }</pre>
 */
public final class Transforms {

    private Transforms() {}

    /** Compose multiple transforms into one. */
    public static UnaryOperator<GradTensor> compose(UnaryOperator<GradTensor>... ops) {
        return t -> { for (var op : ops) t = op.apply(t); return t; };
    }

    // ── Spatial ──────────────────────────────────────────────────────────

    /**
     * Random horizontal flip with probability {@code p}.
     * Input: {@code [C, H, W]} or {@code [N, C, H, W]}.
     */
    public static UnaryOperator<GradTensor> randomHorizontalFlip(float p) {
        return t -> {
            if (Math.random() >= p) return t;
            long[] s = t.shape();
            int W = (int) s[s.length - 1];
            int H = (int) s[s.length - 2];
            int C = (int) s[s.length - 3];
            int N = s.length == 4 ? (int) s[0] : 1;
            float[] d = t.data().clone();
            for (int n = 0; n < N; n++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W / 2; w++) {
                            int base = n * C * H * W + c * H * W + h * W;
                            float tmp = d[base + w]; d[base + w] = d[base + W - 1 - w]; d[base + W - 1 - w] = tmp;
                        }
            return GradTensor.of(d, t.shape());
        };
    }

    /**
     * Random vertical flip with probability {@code p}.
     */
    public static UnaryOperator<GradTensor> randomVerticalFlip(float p) {
        return t -> {
            if (Math.random() >= p) return t;
            long[] s = t.shape();
            int W = (int) s[s.length - 1], H = (int) s[s.length - 2], C = (int) s[s.length - 3];
            int N = s.length == 4 ? (int) s[0] : 1;
            float[] d = t.data().clone();
            for (int n = 0; n < N; n++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < H / 2; h++) {
                        int top = n*C*H*W + c*H*W + h*W, bot = n*C*H*W + c*H*W + (H-1-h)*W;
                        for (int w = 0; w < W; w++) { float tmp = d[top+w]; d[top+w] = d[bot+w]; d[bot+w] = tmp; }
                    }
            return GradTensor.of(d, t.shape());
        };
    }

    /**
     * Random crop: pads then crops to {@code [cropH, cropW]}.
     * Input: {@code [C, H, W]}.
     */
    public static UnaryOperator<GradTensor> randomCrop(int cropH, int cropW, int pad) {
        return t -> {
            long[] s = t.shape();
            int C = (int) s[0], H = (int) s[1], W = (int) s[2];
            int pH = H + 2 * pad, pW = W + 2 * pad;

            // Pad
            float[] padded = new float[C * pH * pW];
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    System.arraycopy(t.data(), c*H*W + h*W, padded, c*pH*pW + (h+pad)*pW + pad, W);

            // Random crop offset
            int oh = (int) (Math.random() * (pH - cropH));
            int ow = (int) (Math.random() * (pW - cropW));
            float[] out = new float[C * cropH * cropW];
            for (int c = 0; c < C; c++)
                for (int h = 0; h < cropH; h++)
                    System.arraycopy(padded, c*pH*pW + (oh+h)*pW + ow, out, c*cropH*cropW + h*cropW, cropW);

            return GradTensor.of(out, C, cropH, cropW);
        };
    }

    // ── Photometric ──────────────────────────────────────────────────────

    /**
     * Normalize per-channel: {@code (x - mean) / std}.
     * Input: {@code [C, H, W]}, mean/std length = C.
     */
    public static UnaryOperator<GradTensor> normalize(float[] mean, float[] std) {
        return t -> {
            long[] s = t.shape();
            int C = (int) s[0], HW = (int) (t.numel() / C);
            float[] d = t.data().clone();
            for (int c = 0; c < C; c++) {
                int base = c * HW;
                for (int i = 0; i < HW; i++) d[base + i] = (d[base + i] - mean[c]) / std[c];
            }
            return GradTensor.of(d, s);
        };
    }

    /**
     * Random brightness/contrast jitter.
     * @param brightness max delta (e.g. 0.2 → ±20%)
     * @param contrast   max factor (e.g. 0.2 → ×[0.8, 1.2])
     */
    public static UnaryOperator<GradTensor> colorJitter(float brightness, float contrast) {
        return t -> {
            float[] d = t.data().clone();
            float bDelta = (float) (Math.random() * 2 - 1) * brightness;
            float cFactor = 1f + (float) (Math.random() * 2 - 1) * contrast;
            for (int i = 0; i < d.length; i++)
                d[i] = Math.min(1f, Math.max(0f, d[i] * cFactor + bDelta));
            return GradTensor.of(d, t.shape());
        };
    }

    // ── Mixup ────────────────────────────────────────────────────────────

    /**
     * Mixup augmentation: blends two samples with Beta(alpha, alpha) weight.
     * Returns mixed input and the lambda used (for loss mixing).
     */
    public static float[] mixup(GradTensor x1, GradTensor x2, float alpha) {
        // Sample lambda from Beta(alpha, alpha) via Gamma approximation
        float lam = sampleBeta(alpha);
        float[] d1 = x1.data(), d2 = x2.data();
        float[] out = new float[d1.length];
        for (int i = 0; i < d1.length; i++) out[i] = lam * d1[i] + (1 - lam) * d2[i];
        return new float[]{lam}; // caller uses lam for label mixing
    }

    public static GradTensor mixupTensor(GradTensor x1, GradTensor x2, float lam) {
        float[] d1 = x1.data(), d2 = x2.data(), out = new float[d1.length];
        for (int i = 0; i < d1.length; i++) out[i] = lam * d1[i] + (1 - lam) * d2[i];
        return GradTensor.of(out, x1.shape());
    }

    private static float sampleBeta(float alpha) {
        // Approximation: Beta(a,a) ≈ clamp(Normal(0.5, 1/(4*a)), 0, 1)
        float std = (float) (1.0 / (4 * alpha));
        float v = 0.5f + (float) (new java.util.Random().nextGaussian() * std);
        return Math.min(1f, Math.max(0f, v));
    }
}
