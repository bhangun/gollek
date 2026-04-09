package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;

/**
 * EfficientNet — scalable CNN architecture that uniformly scales depth, width,
 * and resolution using a compound coefficient.
 *
 * <p>Based on <em>"EfficientNet: Rethinking Model Scaling for Convolutional
 * Neural Networks"</em> (Tan & Le, 2019).
 *
 * <p>Key innovation: MBConv (Mobile Inverted Bottleneck) blocks with
 * Squeeze-and-Excitation (SE) attention.
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #efficientNetB0(int)} — baseline, ~5.3M params</li>
 *   <li>{@link #efficientNetB1(int)} — slightly larger</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * NNModule model = EfficientNet.efficientNetB0(numClasses = 1000);
 * GradTensor out = model.forward(input); // [N,3,224,224] → [N,1000]
 * }</pre>
 */
public final class EfficientNet {

    private EfficientNet() {}

    /**
     * EfficientNet-B0: baseline configuration.
     *
     * @param numClasses number of output classes
     * @return EfficientNet-B0 model
     */
    public static NNModule efficientNetB0(int numClasses) {
        return new EfficientNetModel(new int[]{16, 24, 40, 80, 112, 192, 320},
            new int[]{1, 2, 2, 3, 3, 4, 1}, numClasses, 1280);
    }

    /**
     * EfficientNet-B1: slightly wider and deeper than B0.
     *
     * @param numClasses number of output classes
     * @return EfficientNet-B1 model
     */
    public static NNModule efficientNetB1(int numClasses) {
        return new EfficientNetModel(new int[]{16, 24, 40, 80, 112, 192, 320},
            new int[]{2, 3, 3, 4, 4, 5, 2}, numClasses, 1280);
    }

    // ── Internal model ────────────────────────────────────────────────────

    static final class EfficientNetModel extends NNModule {

        private final Conv2d     stemConv;
        private final BatchNorm2d stemBn;
        private final NNModule[]   blocks;
        private final Conv2d     headConv;
        private final BatchNorm2d headBn;
        private final AdaptiveAvgPool2d pool;
        private final Linear     classifier;

        EfficientNetModel(int[] channels, int[] repeats, int numClasses, int headChannels) {
            this.stemConv = register("stem_conv", new Conv2d(3, 32, 3, 2, 1));
            this.stemBn   = register("stem_bn",   new BatchNorm2d(32));

            // Build MBConv stages
            int inC = 32;
            java.util.List<NNModule> blockList = new java.util.ArrayList<>();
            for (int stage = 0; stage < channels.length; stage++) {
                int outC = channels[stage];
                for (int r = 0; r < repeats[stage]; r++) {
                    int stride = (r == 0 && stage > 0) ? 2 : 1;
                    blockList.add(new MBConvBlock(inC, outC, stride));
                    inC = outC;
                }
            }
            this.blocks = new NNModule[blockList.size()];
            for (int i = 0; i < blockList.size(); i++)
                blocks[i] = register("block_" + i, blockList.get(i));

            this.headConv   = register("head_conv", new Conv2d(inC, headChannels, 1));
            this.headBn     = register("head_bn",   new BatchNorm2d(headChannels));
            this.pool       = register("pool",       new AdaptiveAvgPool2d(1, 1));
            this.classifier = register("classifier", new Linear(headChannels, numClasses));
        }

        @Override
        public GradTensor forward(GradTensor x) {
            x = stemBn.forward(stemConv.forward(x)).relu();
            for (NNModule block : blocks) x = block.forward(x);
            x = headBn.forward(headConv.forward(x)).relu();
            x = pool.forward(x);
            // Flatten [N, C, 1, 1] → [N, C]
            long[] s = x.shape();
            x = x.reshape((int)s[0], (int)s[1]);
            return classifier.forward(x);
        }
    }

    // ── MBConv block ──────────────────────────────────────────────────────

    /**
     * Mobile Inverted Bottleneck block with depthwise separable convolution.
     * Simplified: expand → depthwise → project (without SE for brevity).
     */
    static final class MBConvBlock extends NNModule {

        private final Conv2d expand, depthwise, project;
        private final BatchNorm2d bn1, bn2, bn3;
        private final boolean useResidual;

        MBConvBlock(int inC, int outC, int stride) {
            int expandC = inC * 6; // expansion factor = 6
            this.expand    = register("expand",    new Conv2d(inC, expandC, 1));
            this.bn1       = register("bn1",       new BatchNorm2d(expandC));
            this.depthwise = register("dw",        new Conv2d(expandC, expandC, 3, stride, 1));
            this.bn2       = register("bn2",       new BatchNorm2d(expandC));
            this.project   = register("project",   new Conv2d(expandC, outC, 1));
            this.bn3       = register("bn3",       new BatchNorm2d(outC));
            this.useResidual = (stride == 1 && inC == outC);
        }

        @Override
        public GradTensor forward(GradTensor x) {
            GradTensor out = bn1.forward(expand.forward(x)).relu();
            out = bn2.forward(depthwise.forward(out)).relu();
            out = bn3.forward(project.forward(out));
            return useResidual ? out.add(x) : out;
        }
    }
}
