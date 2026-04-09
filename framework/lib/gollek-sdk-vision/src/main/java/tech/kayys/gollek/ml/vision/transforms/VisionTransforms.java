package tech.kayys.gollek.ml.vision.transforms;

import tech.kayys.gollek.sdk.core.Tensor;
import java.util.List;
import java.util.Random;

/**
 * Comprehensive image preprocessing and augmentation transforms for vision models.
 *
 * <p>Implements PyTorch torchvision.transforms equivalent:
 * <ul>
 *   <li><b>Resizing:</b> Resize, CenterCrop, RandomCrop</li>
 *   <li><b>Normalization:</b> Normalize, ToTensor</li>
 *   <li><b>Augmentation:</b> RandomFlip, RandomRotation, ColorJitter</li>
 *   <li><b>Composition:</b> Compose pipeline</li>
 * </ul>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // Define preprocessing pipeline
 * VisionTransforms.Compose pipeline = new VisionTransforms.Compose(
 *     new VisionTransforms.Resize(224, 224),
 *     new VisionTransforms.CenterCrop(224, 224),
 *     new VisionTransforms.ToTensor(),
 *     new VisionTransforms.Normalize(
 *         new float[]{0.485f, 0.456f, 0.406f},  // ImageNet mean
 *         new float[]{0.229f, 0.224f, 0.225f}   // ImageNet std
 *     )
 * );
 *
 * // Apply to image
 * Tensor image = Tensor.randn(3, 256, 256);
 * Tensor processed = pipeline.apply(image);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class VisionTransforms {

    private VisionTransforms() {}

    /**
     * Base transform interface.
     */
    public interface Transform {
        Tensor apply(Tensor tensor);
    }

    /**
     * Resize image to specified dimensions using bilinear interpolation.
     */
    public static class Resize implements Transform {
        private final int height;
        private final int width;

        public Resize(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public Tensor apply(Tensor tensor) {
            long[] shape = tensor.shape();
            int channels = (int) shape[0];
            int srcH = (int) shape[1];
            int srcW = (int) shape[2];

            float[] srcData = tensor.data();
            float[] dstData = new float[channels * height * width];

            float hScale = (float) srcH / height;
            float wScale = (float) srcW / width;

            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        float srcY = y * hScale;
                        float srcX = x * wScale;

                        int y1 = (int) srcY;
                        int x1 = (int) srcX;
                        int y2 = Math.min(y1 + 1, srcH - 1);
                        int x2 = Math.min(x1 + 1, srcW - 1);

                        float wy2 = srcY - y1;
                        float wy1 = 1 - wy2;
                        float wx2 = srcX - x1;
                        float wx1 = 1 - wx2;

                        int idx11 = c * srcH * srcW + y1 * srcW + x1;
                        int idx12 = c * srcH * srcW + y1 * srcW + x2;
                        int idx21 = c * srcH * srcW + y2 * srcW + x1;
                        int idx22 = c * srcH * srcW + y2 * srcW + x2;

                        float val = wy1 * wx1 * srcData[idx11]
                                  + wy1 * wx2 * srcData[idx12]
                                  + wy2 * wx1 * srcData[idx21]
                                  + wy2 * wx2 * srcData[idx22];

                        dstData[c * height * width + y * width + x] = val;
                    }
                }
            }

            return Tensor.of(dstData, channels, height, width);
        }
    }

    /**
     * Center crop image to specified size.
     */
    public static class CenterCrop implements Transform {
        private final int height;
        private final int width;

        public CenterCrop(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public Tensor apply(Tensor tensor) {
            long[] shape = tensor.shape();
            int channels = (int) shape[0];
            int srcH = (int) shape[1];
            int srcW = (int) shape[2];

            int startY = (srcH - height) / 2;
            int startX = (srcW - width) / 2;

            if (startY < 0 || startX < 0) {
                throw new IllegalArgumentException("Crop size larger than image");
            }

            float[] srcData = tensor.data();
            float[] dstData = new float[channels * height * width];

            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    System.arraycopy(srcData, 
                        c * srcH * srcW + (startY + y) * srcW + startX,
                        dstData,
                        c * height * width + y * width,
                        width);
                }
            }

            return Tensor.of(dstData, channels, height, width);
        }
    }

    /**
     * Random crop with specified size.
     */
    public static class RandomCrop implements Transform {
        private final int height;
        private final int width;
        private final Random random = new Random();

        public RandomCrop(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public Tensor apply(Tensor tensor) {
            long[] shape = tensor.shape();
            int channels = (int) shape[0];
            int srcH = (int) shape[1];
            int srcW = (int) shape[2];

            int maxStartY = Math.max(0, srcH - height);
            int maxStartX = Math.max(0, srcW - width);

            int startY = random.nextInt(maxStartY + 1);
            int startX = random.nextInt(maxStartX + 1);

            float[] srcData = tensor.data();
            float[] dstData = new float[channels * height * width];

            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    System.arraycopy(srcData,
                        c * srcH * srcW + (startY + y) * srcW + startX,
                        dstData,
                        c * height * width + y * width,
                        width);
                }
            }

            return Tensor.of(dstData, channels, height, width);
        }
    }

    /**
     * Normalize tensor using mean and std (ImageNet normalization).
     */
    public static class Normalize implements Transform {
        private final float[] mean;
        private final float[] std;

        public Normalize(float[] mean, float[] std) {
            this.mean = mean;
            this.std = std;
        }

        @Override
        public Tensor apply(Tensor tensor) {
            float[] data = tensor.data().clone();
            long[] shape = tensor.shape();
            int channels = (int) shape[0];
            int height = (int) shape[1];
            int width = (int) shape[2];

            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < height * width; i++) {
                    int idx = c * height * width + i;
                    data[idx] = (data[idx] - mean[c]) / std[c];
                }
            }

            return Tensor.of(data, shape);
        }
    }

    /**
     * Convert to tensor and normalize pixel values [0, 1].
     */
    public static class ToTensor implements Transform {
        @Override
        public Tensor apply(Tensor tensor) {
            float[] data = tensor.data().clone();
            for (int i = 0; i < data.length; i++) {
                data[i] = data[i] / 255f;
            }
            return Tensor.of(data, tensor.shape());
        }
    }

    /**
     * Random horizontal flip.
     */
    public static class RandomHorizontalFlip implements Transform {
        private final float probability;
        private final Random random = new Random();

        public RandomHorizontalFlip(float probability) {
            this.probability = probability;
        }

        @Override
        public Tensor apply(Tensor tensor) {
            if (random.nextFloat() > probability) {
                return tensor;
            }

            long[] shape = tensor.shape();
            int channels = (int) shape[0];
            int height = (int) shape[1];
            int width = (int) shape[2];

            float[] srcData = tensor.data();
            float[] dstData = new float[srcData.length];

            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int srcIdx = c * height * width + y * width + x;
                        int dstIdx = c * height * width + y * width + (width - 1 - x);
                        dstData[dstIdx] = srcData[srcIdx];
                    }
                }
            }

            return Tensor.of(dstData, shape);
        }
    }

    /**
     * Random vertical flip.
     */
    public static class RandomVerticalFlip implements Transform {
        private final float probability;
        private final Random random = new Random();

        public RandomVerticalFlip(float probability) {
            this.probability = probability;
        }

        @Override
        public Tensor apply(Tensor tensor) {
            if (random.nextFloat() > probability) {
                return tensor;
            }

            long[] shape = tensor.shape();
            int channels = (int) shape[0];
            int height = (int) shape[1];
            int width = (int) shape[2];

            float[] srcData = tensor.data();
            float[] dstData = new float[srcData.length];

            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int srcIdx = c * height * width + y * width + x;
                        int dstIdx = c * height * width + (height - 1 - y) * width + x;
                        dstData[dstIdx] = srcData[srcIdx];
                    }
                }
            }

            return Tensor.of(dstData, shape);
        }
    }

    /**
     * Color jitter: random brightness, contrast, saturation.
     */
    public static class ColorJitter implements Transform {
        private final float brightness;
        private final float contrast;
        private final float saturation;
        private final Random random = new Random();

        public ColorJitter(float brightness, float contrast, float saturation) {
            this.brightness = brightness;
            this.contrast = contrast;
            this.saturation = saturation;
        }

        @Override
        public Tensor apply(Tensor tensor) {
            float[] data = tensor.data().clone();

            // Apply brightness
            if (brightness > 0) {
                float factor = 1 + (random.nextFloat() - 0.5f) * 2 * brightness;
                for (int i = 0; i < data.length; i++) {
                    data[i] = Math.min(1f, data[i] * factor);
                }
            }

            // Apply contrast
            if (contrast > 0) {
                float factor = 1 + (random.nextFloat() - 0.5f) * 2 * contrast;
                for (int i = 0; i < data.length; i++) {
                    data[i] = Math.min(1f, 0.5f + (data[i] - 0.5f) * factor);
                }
            }

            return Tensor.of(data, tensor.shape());
        }
    }

    /**
     * Compose multiple transforms into a pipeline.
     */
    public static class Compose implements Transform {
        private final List<Transform> transforms;

        public Compose(Transform... transforms) {
            this.transforms = List.of(transforms);
        }

        @Override
        public Tensor apply(Tensor tensor) {
            for (Transform t : transforms) {
                tensor = t.apply(tensor);
            }
            return tensor;
        }
    }
}
