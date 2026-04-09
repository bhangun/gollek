package tech.kayys.gollek.sdk.vision.transforms;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Image transforms for preprocessing and augmentation.
 * All transforms produce {@code [C, H, W]} float tensors in [0, 1].
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Transform transform = Transform.Compose.of(
 *     Transform.Resize.of(256),
 *     Transform.CenterCrop.of(224),
 *     Transform.ToTensor.of(),
 *     Transform.Normalize.imagenet()
 * );
 * GradTensor x = transform.apply(image);
 * }</pre>
 */
public interface Transform {

    /**
     * Applies this transform to an image.
     *
     * @param image input image (may be null for tensor-only transforms in a Compose chain)
     * @return transformed tensor {@code [C, H, W]}
     */
    GradTensor apply(BufferedImage image);

    // ── Resize ────────────────────────────────────────────────────────────

    /** Resizes the shorter side to {@code size}, preserving aspect ratio. */
    class Resize implements Transform {
        private final int size;

        private Resize(int size) { this.size = size; }
        public static Resize of(int size) { return new Resize(size); }

        @Override
        public GradTensor apply(BufferedImage image) {
            if (image == null) return GradTensor.zeros(3, size, size);
            int W = image.getWidth(), H = image.getHeight();
            int newW, newH;
            if (W < H) { newW = size; newH = H * size / W; }
            else       { newH = size; newW = W * size / H; }
            BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resized.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(image, 0, 0, newW, newH, null);
            g.dispose();
            return ToTensor.of().apply(resized);
        }
    }

    // ── CenterCrop ────────────────────────────────────────────────────────

    /** Crops the center {@code size × size} region. */
    class CenterCrop implements Transform {
        private final int size;

        private CenterCrop(int size) { this.size = size; }
        public static CenterCrop of(int size) { return new CenterCrop(size); }

        @Override
        public GradTensor apply(BufferedImage image) {
            if (image == null) return GradTensor.zeros(3, size, size);
            int W = image.getWidth(), H = image.getHeight();
            int x = Math.max(0, (W - size) / 2), y = Math.max(0, (H - size) / 2);
            int cropW = Math.min(size, W), cropH = Math.min(size, H);
            return ToTensor.of().apply(image.getSubimage(x, y, cropW, cropH));
        }
    }

    // ── ToTensor ──────────────────────────────────────────────────────────

    /** Converts a {@link BufferedImage} to a {@code [3, H, W]} float tensor in [0, 1]. */
    class ToTensor implements Transform {
        private static final ToTensor INSTANCE = new ToTensor();
        private ToTensor() {}
        public static ToTensor of() { return INSTANCE; }

        @Override
        public GradTensor apply(BufferedImage image) {
            if (image == null) return GradTensor.zeros(3, 1, 1);
            int W = image.getWidth(), H = image.getHeight();
            float[] data = new float[3 * H * W];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    int rgb = image.getRGB(x, y);
                    int idx = y * W + x;
                    data[idx]           = ((rgb >> 16) & 0xFF) / 255f; // R
                    data[H*W + idx]     = ((rgb >>  8) & 0xFF) / 255f; // G
                    data[2*H*W + idx]   = ( rgb        & 0xFF) / 255f; // B
                }
            return GradTensor.of(data, 3, H, W);
        }
    }

    // ── Normalize ─────────────────────────────────────────────────────────

    /**
     * Per-channel normalization: {@code (x - mean) / std}.
     * Applied to a tensor (not an image); use after {@link ToTensor}.
     */
    class Normalize implements Transform {
        private final float[] mean, std;

        private Normalize(float[] mean, float[] std) { this.mean = mean; this.std = std; }
        public static Normalize of(float[] mean, float[] std) { return new Normalize(mean, std); }

        /** ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]. */
        public static Normalize imagenet() {
            return new Normalize(
                new float[]{0.485f, 0.456f, 0.406f},
                new float[]{0.229f, 0.224f, 0.225f});
        }

        /** Applies normalization to a tensor (image parameter is ignored). */
        @Override
        public GradTensor apply(BufferedImage image) {
            throw new UnsupportedOperationException("Normalize requires a tensor; use applyToTensor(GradTensor)");
        }

        /**
         * Normalizes a {@code [C, H, W]} tensor.
         *
         * @param t input tensor
         * @return normalized tensor
         */
        public GradTensor applyToTensor(GradTensor t) {
            long[] s = t.shape();
            int C = (int)s[0], HW = (int)(t.numel() / C);
            float[] d = t.data().clone();
            for (int c = 0; c < C && c < mean.length; c++) {
                int base = c * HW;
                for (int i = 0; i < HW; i++) d[base+i] = (d[base+i] - mean[c]) / std[c];
            }
            return GradTensor.of(d, s);
        }
    }

    // ── Compose ───────────────────────────────────────────────────────────

    /**
     * Chains multiple transforms: first applies the initial transform to the image,
     * then applies subsequent {@link Normalize} transforms to the resulting tensor.
     */
    class Compose implements Transform {
        private final Transform[] transforms;

        private Compose(Transform... transforms) { this.transforms = transforms; }
        public static Compose of(Transform... transforms) { return new Compose(transforms); }

        @Override
        public GradTensor apply(BufferedImage image) {
            GradTensor result = null;
            for (Transform t : transforms) {
                if (result == null) {
                    result = t.apply(image);
                } else if (t instanceof Normalize n) {
                    result = n.applyToTensor(result);
                }
                // Other tensor-to-tensor transforms can be added here
            }
            return result;
        }
    }
}
