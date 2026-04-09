package tech.kayys.gollek.sdk.cnn.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Upsample Layer.
 *
 * <p>Upsamples a given multi-channel 2D/3D spatial data.
 * Supports nearest neighbor and bilinear/trilinear interpolation.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Upsample up = new Upsample(2, "nearest");  // scale_factor=2, mode="nearest"
 * GradTensor x = GradTensor.randn(2, 3, 28, 28);
 * GradTensor y = up.forward(x);  // Shape: [2, 3, 56, 56]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class Upsample {

    private final double scaleFactor;
    private final String mode;  // "nearest", "linear", "bilinear", "trilinear"
    private final boolean alignCorners;

    /**
     * Create an upsample layer.
     *
     * @param scaleFactor   scale factor for upsampling (must be > 1)
     * @param mode          interpolation mode: "nearest", "linear", "bilinear", "trilinear"
     * @param alignCorners  if true, align corners when mode is bilinear/trilinear
     */
    public Upsample(double scaleFactor, String mode, boolean alignCorners) {
        if (scaleFactor <= 1.0) {
            throw new IllegalArgumentException("scaleFactor must be > 1.0, got: " + scaleFactor);
        }
        if (!mode.matches("nearest|linear|bilinear|trilinear")) {
            throw new IllegalArgumentException("Invalid mode: " + mode);
        }
        this.scaleFactor = scaleFactor;
        this.mode = mode;
        this.alignCorners = alignCorners;
    }

    /**
     * Create Upsample with default alignCorners=false.
     */
    public Upsample(double scaleFactor, String mode) {
        this(scaleFactor, mode, false);
    }

    /**
     * Forward pass (nearest neighbor).
     *
     * @param input input tensor of shape [N, C, H, W] or [N, C, D, H, W]
     * @return output tensor with upsampled spatial dimensions
     */
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();

        if (shape.length == 4) {
            return upsample2d(input);
        } else if (shape.length == 5) {
            return upsample3d(input);
        } else {
            throw new IllegalArgumentException("Expected 4D or 5D tensor, got " + shape.length + "D");
        }
    }

    /**
     * Upsample 2D input.
     */
    private GradTensor upsample2d(GradTensor input) {
        long[] shape = input.shape();
        int N = (int) shape[0];
        int C = (int) shape[1];
        int H = (int) shape[2];
        int W = (int) shape[3];

        int outH = (int) (H * scaleFactor);
        int outW = (int) (W * scaleFactor);

        float[] inData = input.data();
        float[] outData = new float[N * C * outH * outW];

        if ("nearest".equals(mode)) {
            upsampleNearest2d(inData, outData, N, C, H, W, outH, outW);
        } else if ("bilinear".equals(mode)) {
            upsampleBilinear2d(inData, outData, N, C, H, W, outH, outW);
        } else {
            throw new UnsupportedOperationException("Mode " + mode + " not yet implemented for 2D");
        }

        return GradTensor.of(outData, N, C, outH, outW);
    }

    /**
     * Upsample 3D input.
     */
    private GradTensor upsample3d(GradTensor input) {
        long[] shape = input.shape();
        int N = (int) shape[0];
        int C = (int) shape[1];
        int D = (int) shape[2];
        int H = (int) shape[3];
        int W = (int) shape[4];

        int outD = (int) (D * scaleFactor);
        int outH = (int) (H * scaleFactor);
        int outW = (int) (W * scaleFactor);

        float[] inData = input.data();
        float[] outData = new float[N * C * outD * outH * outW];

        if ("nearest".equals(mode)) {
            upsampleNearest3d(inData, outData, N, C, D, H, W, outD, outH, outW);
        } else if ("trilinear".equals(mode)) {
            upsampleTrilinear3d(inData, outData, N, C, D, H, W, outD, outH, outW);
        } else {
            throw new UnsupportedOperationException("Mode " + mode + " not yet implemented for 3D");
        }

        return GradTensor.of(outData, N, C, outD, outH, outW);
    }

    /**
     * Nearest neighbor upsampling for 2D.
     */
    private void upsampleNearest2d(float[] inData, float[] outData,
                                   int N, int C, int H, int W, int outH, int outW) {
        int strideH = (int) (H / (double) outH * scaleFactor);
        int strideW = (int) (W / (double) outW * scaleFactor);

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        int inH = (int) (h / scaleFactor);
                        int inW = (int) (w / scaleFactor);

                        int inIdx = ((n * C + c) * H + inH) * W + inW;
                        int outIdx = ((n * C + c) * outH + h) * outW + w;

                        outData[outIdx] = inData[inIdx];
                    }
                }
            }
        }
    }

    /**
     * Bilinear upsampling for 2D.
     */
    private void upsampleBilinear2d(float[] inData, float[] outData,
                                    int N, int C, int H, int W, int outH, int outW) {
        double scaleH = (H - 1.0) / (outH - 1.0);
        double scaleW = (W - 1.0) / (outW - 1.0);

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        double srcH = h * scaleH;
                        double srcW = w * scaleW;

                        int h0 = (int) srcH;
                        int h1 = Math.min(h0 + 1, H - 1);
                        int w0 = (int) srcW;
                        int w1 = Math.min(w0 + 1, W - 1);

                        double dh = srcH - h0;
                        double dw = srcW - w0;

                        float v00 = inData[((n * C + c) * H + h0) * W + w0];
                        float v01 = inData[((n * C + c) * H + h0) * W + w1];
                        float v10 = inData[((n * C + c) * H + h1) * W + w0];
                        float v11 = inData[((n * C + c) * H + h1) * W + w1];

                        float val = (float) ((1 - dh) * (1 - dw) * v00 +
                                           (1 - dh) * dw * v01 +
                                           dh * (1 - dw) * v10 +
                                           dh * dw * v11);

                        outData[((n * C + c) * outH + h) * outW + w] = val;
                    }
                }
            }
        }
    }

    /**
     * Nearest neighbor upsampling for 3D.
     */
    private void upsampleNearest3d(float[] inData, float[] outData,
                                   int N, int C, int D, int H, int W,
                                   int outD, int outH, int outW) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int d = 0; d < outD; d++) {
                    for (int h = 0; h < outH; h++) {
                        for (int w = 0; w < outW; w++) {
                            int inD = (int) (d / scaleFactor);
                            int inH = (int) (h / scaleFactor);
                            int inW = (int) (w / scaleFactor);

                            int inIdx = (((n * C + c) * D + inD) * H + inH) * W + inW;
                            int outIdx = (((n * C + c) * outD + d) * outH + h) * outW + w;

                            outData[outIdx] = inData[inIdx];
                        }
                    }
                }
            }
        }
    }

    /**
     * Trilinear upsampling for 3D.
     */
    private void upsampleTrilinear3d(float[] inData, float[] outData,
                                     int N, int C, int D, int H, int W,
                                     int outD, int outH, int outW) {
        double scaleD = (D - 1.0) / (outD - 1.0);
        double scaleH = (H - 1.0) / (outH - 1.0);
        double scaleW = (W - 1.0) / (outW - 1.0);

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int d = 0; d < outD; d++) {
                    for (int h = 0; h < outH; h++) {
                        for (int w = 0; w < outW; w++) {
                            double srcD = d * scaleD;
                            double srcH = h * scaleH;
                            double srcW = w * scaleW;

                            int d0 = (int) srcD;
                            int h0 = (int) srcH;
                            int w0 = (int) srcW;

                            int d1 = Math.min(d0 + 1, D - 1);
                            int h1 = Math.min(h0 + 1, H - 1);
                            int w1 = Math.min(w0 + 1, W - 1);

                            double dd = srcD - d0;
                            double dh = srcH - h0;
                            double dw = srcW - w0;

                            float v = 0;
                            for (int di = 0; di < 2; di++) {
                                for (int hi = 0; hi < 2; hi++) {
                                    for (int wi = 0; wi < 2; wi++) {
                                        int cd = (di == 0) ? d0 : d1;
                                        int ch = (hi == 0) ? h0 : h1;
                                        int cw = (wi == 0) ? w0 : w1;

                                        float val = inData[(((n * C + c) * D + cd) * H + ch) * W + cw];
                                        double weight = ((di == 0) ? (1 - dd) : dd) *
                                                       ((hi == 0) ? (1 - dh) : dh) *
                                                       ((wi == 0) ? (1 - dw) : dw);

                                        v += (float) (weight * val);
                                    }
                                }
                            }

                            outData[(((n * C + c) * outD + d) * outH + h) * outW + w] = v;
                        }
                    }
                }
            }
        }
    }

    /**
     * Get scale factor.
     */
    public double getScaleFactor() {
        return scaleFactor;
    }

    /**
     * Get interpolation mode.
     */
    public String getMode() {
        return mode;
    }

    @Override
    public String toString() {
        return String.format("Upsample(scale_factor=%.1f, mode=%s)", scaleFactor, mode);
    }
}
