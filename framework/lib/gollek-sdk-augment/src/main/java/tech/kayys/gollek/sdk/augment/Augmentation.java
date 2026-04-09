package tech.kayys.gollek.sdk.augment;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Random;

/**
 * Data augmentation transforms for training robust vision models.
 * All transforms operate on {@code [C, H, W]} float tensors in [0, 1].
 */
public interface Augmentation {

    /**
     * Apply augmentation to a tensor.
     *
     * @param input input tensor {@code [C, H, W]}
     * @param rng   random number generator for reproducibility
     * @return augmented tensor
     */
    GradTensor apply(GradTensor input, Random rng);

    // ── Horizontal flip ───────────────────────────────────────────────────

    /** Random horizontal flip with given probability. */
    class RandomHorizontalFlip implements Augmentation {
        private final double probability;

        private RandomHorizontalFlip(double probability) { this.probability = probability; }

        public static RandomHorizontalFlip of(double probability) { return new RandomHorizontalFlip(probability); }
        public static RandomHorizontalFlip create()               { return new RandomHorizontalFlip(0.5); }

        @Override
        public GradTensor apply(GradTensor input, Random rng) {
            if (rng.nextDouble() >= probability) return input;
            long[] s = input.shape();
            int C = (int)s[0], H = (int)s[1], W = (int)s[2];
            float[] d = input.data().clone();
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W / 2; w++) {
                        int a = c*H*W + h*W + w, b = c*H*W + h*W + (W-1-w);
                        float tmp = d[a]; d[a] = d[b]; d[b] = tmp;
                    }
            return GradTensor.of(d, s);
        }
    }

    // ── Vertical flip ─────────────────────────────────────────────────────

    /** Random vertical flip with given probability. */
    class RandomVerticalFlip implements Augmentation {
        private final double probability;

        private RandomVerticalFlip(double probability) { this.probability = probability; }
        public static RandomVerticalFlip of(double probability) { return new RandomVerticalFlip(probability); }

        @Override
        public GradTensor apply(GradTensor input, Random rng) {
            if (rng.nextDouble() >= probability) return input;
            long[] s = input.shape();
            int C = (int)s[0], H = (int)s[1], W = (int)s[2];
            float[] d = input.data().clone();
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H / 2; h++) {
                    int top = c*H*W + h*W, bot = c*H*W + (H-1-h)*W;
                    for (int w = 0; w < W; w++) { float tmp = d[top+w]; d[top+w] = d[bot+w]; d[bot+w] = tmp; }
                }
            return GradTensor.of(d, s);
        }
    }

    // ── Random crop ───────────────────────────────────────────────────────

    /** Pads by {@code padding} then crops to {@code size × size}. */
    class RandomCrop implements Augmentation {
        private final int size;
        private final int padding;

        private RandomCrop(int size, int padding) { this.size = size; this.padding = padding; }
        public static RandomCrop of(int size)              { return new RandomCrop(size, size / 8); }
        public static RandomCrop of(int size, int padding) { return new RandomCrop(size, padding); }

        @Override
        public GradTensor apply(GradTensor input, Random rng) {
            long[] s = input.shape();
            int C = (int)s[0], H = (int)s[1], W = (int)s[2];
            int pH = H + 2*padding, pW = W + 2*padding;

            // Pad with zeros
            float[] padded = new float[C * pH * pW];
            float[] src = input.data();
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    System.arraycopy(src, c*H*W + h*W, padded, c*pH*pW + (h+padding)*pW + padding, W);

            // Random crop offset
            int oh = rng.nextInt(Math.max(1, pH - size));
            int ow = rng.nextInt(Math.max(1, pW - size));
            int cropH = Math.min(size, pH - oh), cropW = Math.min(size, pW - ow);
            float[] out = new float[C * cropH * cropW];
            for (int c = 0; c < C; c++)
                for (int h = 0; h < cropH; h++)
                    System.arraycopy(padded, c*pH*pW + (oh+h)*pW + ow, out, c*cropH*cropW + h*cropW, cropW);
            return GradTensor.of(out, C, cropH, cropW);
        }
    }

    // ── Color jitter ──────────────────────────────────────────────────────

    /** Random brightness, contrast, and saturation jitter. */
    class ColorJitter implements Augmentation {
        private final double brightness, contrast, saturation;

        private ColorJitter(double brightness, double contrast, double saturation) {
            this.brightness = brightness; this.contrast = contrast; this.saturation = saturation;
        }
        public static ColorJitter of(double b, double c, double s) { return new ColorJitter(b, c, s); }

        @Override
        public GradTensor apply(GradTensor input, Random rng) {
            float[] d = input.data().clone();
            float bDelta = (float)((rng.nextDouble()*2-1) * brightness);
            float cFactor = 1f + (float)((rng.nextDouble()*2-1) * contrast);
            for (int i = 0; i < d.length; i++)
                d[i] = Math.min(1f, Math.max(0f, d[i] * cFactor + bDelta));
            return GradTensor.of(d, input.shape());
        }
    }

    // ── Random erasing ────────────────────────────────────────────────────

    /** Randomly erases a rectangular patch (cutout regularization). */
    class RandomErasing implements Augmentation {
        private final double probability;
        private final double scale;

        private RandomErasing(double probability, double scale, double ratio) {
            this.probability = probability; this.scale = scale;
        }
        public static RandomErasing of(double probability) { return new RandomErasing(probability, 0.02, 0.3); }

        @Override
        public GradTensor apply(GradTensor input, Random rng) {
            if (rng.nextDouble() >= probability) return input;
            long[] s = input.shape();
            int C = (int)s[0], H = (int)s[1], W = (int)s[2];
            int area = (int)(H * W * scale * (1 + rng.nextDouble()));
            int eh = (int) Math.sqrt(area), ew = area / Math.max(1, eh);
            int oh = rng.nextInt(Math.max(1, H - eh)), ow = rng.nextInt(Math.max(1, W - ew));
            float[] d = input.data().clone();
            for (int c = 0; c < C; c++)
                for (int h = oh; h < Math.min(oh+eh, H); h++)
                    for (int w = ow; w < Math.min(ow+ew, W); w++)
                        d[c*H*W + h*W + w] = 0f;
            return GradTensor.of(d, s);
        }
    }

    // ── Grayscale ─────────────────────────────────────────────────────────

    /** Converts RGB to grayscale with given probability. */
    class RandomGrayscale implements Augmentation {
        private final double probability;

        private RandomGrayscale(double probability) { this.probability = probability; }
        public static RandomGrayscale of(double probability) { return new RandomGrayscale(probability); }

        @Override
        public GradTensor apply(GradTensor input, Random rng) {
            if (rng.nextDouble() >= probability) return input;
            long[] s = input.shape();
            int C = (int)s[0], H = (int)s[1], W = (int)s[2];
            if (C < 3) return input;
            float[] src = input.data(), dst = new float[C * H * W];
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    float gray = 0.299f*src[h*W+w] + 0.587f*src[H*W+h*W+w] + 0.114f*src[2*H*W+h*W+w];
                    for (int c = 0; c < C; c++) dst[c*H*W + h*W + w] = gray;
                }
            return GradTensor.of(dst, s);
        }
    }
}
