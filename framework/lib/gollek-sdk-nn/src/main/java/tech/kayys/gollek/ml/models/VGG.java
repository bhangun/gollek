package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;

/**
 * VGG network family — deep CNNs with small 3×3 convolutions.
 *
 * <p>Based on <em>"Very Deep Convolutional Networks for Large-Scale Image Recognition"</em>
 * (Simonyan & Zisserman, 2014).
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #vgg11(int)} — 8 conv layers, ~133M params</li>
 *   <li>{@link #vgg16(int)} — 13 conv layers, ~138M params</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * NNModule model = VGG.vgg16(numClasses = 1000);
 * GradTensor out = model.forward(input); // [N,3,224,224] → [N,1000]
 * }</pre>
 */
public final class VGG {

    private VGG() {}

    /**
     * VGG-11: 8 convolutional layers.
     *
     * @param numClasses number of output classes
     * @return VGG-11 model
     */
    public static NNModule vgg11(int numClasses) {
        return new VGGModel(new int[][]{
            {64}, {128}, {256, 256}, {512, 512}, {512, 512}
        }, numClasses);
    }

    /**
     * VGG-16: 13 convolutional layers.
     *
     * @param numClasses number of output classes
     * @return VGG-16 model
     */
    public static NNModule vgg16(int numClasses) {
        return new VGGModel(new int[][]{
            {64, 64}, {128, 128}, {256, 256, 256}, {512, 512, 512}, {512, 512, 512}
        }, numClasses);
    }

    // ── Internal model ────────────────────────────────────────────────────

    static final class VGGModel extends NNModule {

        private final NNModule[] features; // conv blocks
        private final AdaptiveAvgPool2d pool;
        private final Linear fc1, fc2, fc3;
        private final Dropout drop;

        VGGModel(int[][] blockChannels, int numClasses) {
            // Build conv feature extractor
            java.util.List<NNModule> layers = new java.util.ArrayList<>();
            int inC = 3;
            for (int[] block : blockChannels) {
                for (int outC : block) {
                    layers.add(new Conv2d(inC, outC, 3, 1, 1));
                    layers.add(new BatchNorm2d(outC));
                    layers.add(new ReLU());
                    inC = outC;
                }
                layers.add(new MaxPool2d(2)); // halve spatial dims
            }
            this.features = new NNModule[layers.size()];
            for (int i = 0; i < layers.size(); i++)
                features[i] = register("feat_" + i, layers.get(i));

            this.pool = register("pool", new AdaptiveAvgPool2d(7, 7));
            this.fc1  = register("fc1",  new Linear(512 * 7 * 7, 4096));
            this.fc2  = register("fc2",  new Linear(4096, 4096));
            this.fc3  = register("fc3",  new Linear(4096, numClasses));
            this.drop = register("drop", new Dropout(0.5f));
        }

        @Override
        public GradTensor forward(GradTensor x) {
            for (NNModule f : features) x = f.forward(x);
            x = pool.forward(x);
            long[] s = x.shape();
            x = x.reshape((int)s[0], (int)(x.numel() / s[0]));
            x = drop.forward(fc1.forward(x).relu());
            x = drop.forward(fc2.forward(x).relu());
            return fc3.forward(x);
        }
    }
}
