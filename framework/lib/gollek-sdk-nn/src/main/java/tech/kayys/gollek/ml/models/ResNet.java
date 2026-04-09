package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;

/**
 * ResNet image classification models — equivalent to {@code torchvision.models.resnet*}.
 *
 * <p>Implements the residual network architecture from
 * <em>"Deep Residual Learning for Image Recognition"</em> (He et al., 2015).
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #resnet18(int)} — 18 layers, ~11M parameters</li>
 *   <li>{@link #resnet34(int)} — 34 layers, ~21M parameters</li>
 *   <li>{@link #resnet50(int)} — 50 layers with bottleneck, ~25M parameters</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * NNModule model = ResNet.resnet18(numClasses = 1000);
 * GradTensor out = model.forward(input); // [B, 3, 224, 224] → [B, 1000]
 * }</pre>
 */
public final class ResNet {

    private ResNet() {}

    /**
     * Constructs a ResNet-18 model.
     *
     * <p>Architecture: {@code [3,64] → 2×[64,64] → 2×[128,128] → 2×[256,256] → 2×[512,512] → FC}
     *
     * @param numClasses number of output classes
     * @return ResNet-18 {@link Module}
     */
    public static NNModule resnet18(int numClasses) {
        return new ResNetModel(new int[]{2, 2, 2, 2}, numClasses, false);
    }

    /**
     * Constructs a ResNet-34 model.
     *
     * @param numClasses number of output classes
     * @return ResNet-34 {@link Module}
     */
    public static NNModule resnet34(int numClasses) {
        return new ResNetModel(new int[]{3, 4, 6, 3}, numClasses, false);
    }

    /**
     * Constructs a ResNet-50 model with bottleneck blocks.
     *
     * @param numClasses number of output classes
     * @return ResNet-50 {@link Module}
     */
    public static NNModule resnet50(int numClasses) {
        return new ResNetModel(new int[]{3, 4, 6, 3}, numClasses, true);
    }

    // ── Internal model ────────────────────────────────────────────────────

    private static final class ResNetModel extends NNModule {

        private final Conv2d   stem;
        private final BatchNorm1d stemBn;
        private final MaxPool2d   stemPool;
        private final NNModule[]    layers;
        private final Linear      fc;
        private final boolean     bottleneck;

        ResNetModel(int[] layerCounts, int numClasses, boolean bottleneck) {
            this.bottleneck = bottleneck;
            int[] channels = bottleneck
                ? new int[]{256, 512, 1024, 2048}
                : new int[]{64,  128, 256,  512};

            this.stem     = register("stem",     new Conv2d(3, 64, 7, 2, 3));
            this.stemBn   = register("stem_bn",  new BatchNorm1d(64));
            this.stemPool = register("stem_pool",new MaxPool2d(3, 2, 1));

            this.layers = new NNModule[4];
            int inC = 64;
            for (int i = 0; i < 4; i++) {
                layers[i] = register("layer" + (i + 1),
                    makeLayer(inC, channels[i], layerCounts[i], i > 0 ? 2 : 1, bottleneck));
                inC = channels[i];
            }
            this.fc = register("fc", new Linear(channels[3], numClasses));
        }

        @Override
        public GradTensor forward(GradTensor x) {
            // Stem: Conv7×7 → BN → ReLU → MaxPool
            x = stemPool.forward(stemBn.forward(stem.forward(x)).relu());

            // Residual stages
            for (NNModule layer : layers) x = layer.forward(x);

            // Global average pool: [B, C, H, W] → [B, C]
            long[] s = x.shape();
            int B = (int) s[0], C = (int) s[1], H = (int) s[2], W = (int) s[3];
            float[] pooled = new float[B * C];
            float[] d = x.data();
            for (int b = 0; b < B; b++)
                for (int c = 0; c < C; c++) {
                    float sum = 0;
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            sum += d[b*C*H*W + c*H*W + h*W + w];
                    pooled[b * C + c] = sum / (H * W);
                }
            return fc.forward(GradTensor.of(pooled, B, C));
        }

        /** Builds one residual stage (sequence of blocks). */
        private NNModule makeLayer(int inC, int outC, int count, int stride, boolean bottleneck) {
            return new Sequential(buildBlocks(inC, outC, count, stride, bottleneck));
        }

        private NNModule[] buildBlocks(int inC, int outC, int count, int stride, boolean bottleneck) {
            NNModule[] blocks = new NNModule[count];
            blocks[0] = bottleneck
                ? new BottleneckBlock(inC, outC, stride)
                : new BasicBlock(inC, outC, stride);
            for (int i = 1; i < count; i++)
                blocks[i] = bottleneck
                    ? new BottleneckBlock(outC, outC, 1)
                    : new BasicBlock(outC, outC, 1);
            return blocks;
        }
    }

    // ── Bottleneck block (ResNet-50+) ─────────────────────────────────────

    /**
     * Bottleneck residual block: 1×1 → 3×3 → 1×1 convolutions.
     * Used in ResNet-50/101/152 for parameter efficiency.
     */
    static final class BottleneckBlock extends NNModule {

        private final Conv2d conv1, conv2, conv3;
        private final BatchNorm1d bn1, bn2, bn3;
        private final Conv2d downsample;
        private final boolean hasDownsample;

        /**
         * @param inChannels  input channel count
         * @param outChannels output channel count (bottleneck width = outChannels / 4)
         * @param stride      stride for the 3×3 conv (2 = downsample spatial dims)
         */
        BottleneckBlock(int inChannels, int outChannels, int stride) {
            int mid = outChannels / 4;
            this.conv1 = register("conv1", new Conv2d(inChannels, mid, 1));
            this.bn1   = register("bn1",   new BatchNorm1d(mid));
            this.conv2 = register("conv2", new Conv2d(mid, mid, 3, stride, 1));
            this.bn2   = register("bn2",   new BatchNorm1d(mid));
            this.conv3 = register("conv3", new Conv2d(mid, outChannels, 1));
            this.bn3   = register("bn3",   new BatchNorm1d(outChannels));
            this.hasDownsample = inChannels != outChannels || stride != 1;
            this.downsample = hasDownsample
                ? register("ds", new Conv2d(inChannels, outChannels, 1, stride, 0))
                : null;
        }

        @Override
        public GradTensor forward(GradTensor x) {
            GradTensor identity = x;
            GradTensor out = bn1.forward(conv1.forward(x)).relu();
            out = bn2.forward(conv2.forward(out)).relu();
            out = bn3.forward(conv3.forward(out));
            if (hasDownsample) identity = downsample.forward(x);
            return out.add(identity).relu();
        }
    }

    // ── Basic block (ResNet-18/34) ─────────────────────────────────────

    /**
     * Basic residual block: 3×3 → 3×3 convolutions.
     * Used in ResNet-18/34 for simplicity.
     */
    static final class BasicBlock extends NNModule {

        private final Conv2d conv1, conv2;
        private final BatchNorm1d bn1, bn2;
        private final Conv2d downsample;
        private final boolean hasDownsample;

        /**
         * @param inChannels  input channel count
         * @param outChannels output channel count
         * @param stride      stride for the first 3×3 conv (2 = downsample spatial dims)
         */
        BasicBlock(int inChannels, int outChannels, int stride) {
            this.conv1 = register("conv1", new Conv2d(inChannels, outChannels, 3, stride, 1));
            this.bn1   = register("bn1",   new BatchNorm1d(outChannels));
            this.conv2 = register("conv2", new Conv2d(outChannels, outChannels, 3, 1, 1));
            this.bn2   = register("bn2",   new BatchNorm1d(outChannels));
            this.hasDownsample = inChannels != outChannels || stride != 1;
            this.downsample = hasDownsample
                ? register("ds", new Conv2d(inChannels, outChannels, 1, stride, 0))
                : null;
        }

        @Override
        public GradTensor forward(GradTensor x) {
            GradTensor identity = x;
            GradTensor out = bn1.forward(conv1.forward(x)).relu();
            out = bn2.forward(conv2.forward(out));
            if (hasDownsample) identity = downsample.forward(x);
            return out.add(identity).relu();
        }
    }
}
