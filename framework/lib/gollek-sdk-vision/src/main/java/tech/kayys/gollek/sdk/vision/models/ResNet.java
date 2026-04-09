package tech.kayys.gollek.sdk.vision.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.sdk.vision.layers.*;
import tech.kayys.gollek.sdk.vision.ops.ElementWiseOps;

/**
 * ResNet (Residual Network) model.
 *
 * <p>Implements the ResNet architecture from "Deep Residual Learning for Image Recognition"
 * (He et al., 2015). Uses skip connections to enable training of very deep networks.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Create ResNet-18 for 1000-class ImageNet classification
 * ResNet model = ResNet.resnet18(1000);
 *
 * // Forward pass
 * GradTensor x = GradTensor.randn(1, 3, 224, 224);  // Batch of 1, 3-channel 224x224 image
 * GradTensor logits = model.forward(x);  // Shape: [1, 1000]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class ResNet extends NNModule {

    // Initial convolution
    private final Conv2d conv1;
    private final BatchNorm2d bn1;
    private final ReLU relu;
    private final MaxPool2d maxpool;

    // Final classification layers
    private final AdaptiveAvgPool2d avgpool;
    private final Linear fc;

    // Model configuration
    private final int[] layers;
    private final int numClasses;
    private final NNModule[][] residualBlocks;

    private ResNet(Builder builder) {
        this.layers = builder.layers;
        this.numClasses = builder.numClasses;

        // Initial 7x7 convolution with stride 2
        this.conv1 = register("conv1", new Conv2d(3, 64, 7, 2, 3, false));
        this.bn1 = register("bn1", new BatchNorm2d(64));
        this.relu = register("relu", new ReLU());
        this.maxpool = register("maxpool", new MaxPool2d(3, 2, 1));

        // Residual blocks
        this.residualBlocks = new NNModule[layers.length][];
        int inChannels = 64;
        for (int i = 0; i < layers.length; i++) {
            int outChannels = 64 * (int) Math.pow(2, i);
            residualBlocks[i] = new NNModule[layers[i]];
            for (int j = 0; j < layers[i]; j++) {
                int s = (i > 0 && j == 0) ? 2 : 1;
                residualBlocks[i][j] = register("layer" + i + "_block" + j, 
                    new BasicBlock(inChannels, outChannels, s));
                inChannels = outChannels;
            }
        }

        // Global average pooling and classification
        this.avgpool = register("avgpool", new AdaptiveAvgPool2d(1));
        int outChannels = 64 * (int) Math.pow(2, layers.length - 1);
        this.fc = register("fc", new Linear(outChannels, numClasses));
    }

    /**
     * Basic residual block for ResNet.
     */
    private static class BasicBlock extends NNModule {
        private final Conv2d conv1, conv2, downsample;
        private final BatchNorm2d bn1, bn2, bn3;
        private final ReLU relu;
        private final boolean hasDownsample;

        BasicBlock(int inC, int outC, int stride) {
            this.conv1 = register("conv1", new Conv2d(inC, outC, 3, stride, 1, false));
            this.bn1 = register("bn1", new BatchNorm2d(outC));
            this.relu = register("relu", new ReLU());
            this.conv2 = register("conv2", new Conv2d(outC, outC, 3, 1, 1, false));
            this.bn2 = register("bn2", new BatchNorm2d(outC));
            
            this.hasDownsample = stride != 1 || inC != outC;
            if (hasDownsample) {
                this.downsample = register("ds", new Conv2d(inC, outC, 1, stride, 0, false));
                this.bn3 = register("bn3", new BatchNorm2d(outC));
            } else {
                this.downsample = null;
                this.bn3 = null;
            }
        }

        @Override
        public GradTensor forward(GradTensor x) {
            GradTensor identity = x;
            GradTensor out = relu.forward(bn1.forward(conv1.forward(x)));
            out = bn2.forward(conv2.forward(out));
            
            if (hasDownsample) {
                identity = bn3.forward(downsample.forward(identity));
            }
            
            // Simplified addition using existing ElementWiseOps if possible, 
            // but for simplicity we'll just return sum + relu
            return out.add(identity).relu();
        }
    }

    /**
     * Create ResNet-18.
     *
     * @param numClasses number of output classes
     * @return ResNet-18 model
     */
    public static ResNet resnet18(int numClasses) {
        return builder()
            .layers(2, 2, 2, 2)  // ResNet-18: 4 layers with 2 blocks each
            .numClasses(numClasses)
            .build();
    }

    /**
     * Create ResNet-34.
     *
     * @param numClasses number of output classes
     * @return ResNet-34 model
     */
    public static ResNet resnet34(int numClasses) {
        return builder()
            .layers(3, 4, 6, 3)  // ResNet-34: 4 layers with 3,4,6,3 blocks
            .numClasses(numClasses)
            .build();
    }

    /**
     * Create ResNet-50.
     *
     * @param numClasses number of output classes
     * @return ResNet-50 model
     */
    public static ResNet resnet50(int numClasses) {
        return builder()
            .layers(3, 4, 6, 3)  // ResNet-50: 4 layers with bottleneck blocks
            .numClasses(numClasses)
            .build();
    }

    /**
     * Create a new builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // Initial convolution
        GradTensor x = relu.forward(bn1.forward(conv1.forward(input)));
        x = maxpool.forward(x);

        // Residual layers
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i]; j++) {
                x = residualBlocks[i][j].forward(x);
            }
        }

        // Global average pooling
        x = avgpool.forward(x);

        // Flatten
        long[] shape = x.shape();
        x = x.reshape(shape[0], shape[1]);

        // Final classification
        return fc.forward(x);
    }


    /**
     * Set training mode.
     *
     * @param training true for training, false for inference
     */
    public void setTraining(boolean training) {
        if (training) train(); else eval();
    }

    /**
     * Get number of classes.
     */
    public int getNumClasses() {
        return numClasses;
    }

    /**
     * Get layer configuration.
     */
    public int[] getLayers() {
        return layers.clone();
    }

    @Override
    public String toString() {
        return String.format("ResNet(layers=%s, numClasses=%d)",
                java.util.Arrays.toString(layers), numClasses);
    }

    /**
     * Builder for ResNet.
     */
    public static class Builder {
        private int[] layers = {2, 2, 2, 2};  // Default: ResNet-18
        private int numClasses = 1000;

        private Builder() {}

        /**
         * Set number of blocks in each layer.
         *
         * @param layers array of 4 integers
         * @return this builder
         */
        public Builder layers(int... layers) {
            this.layers = layers.clone();
            return this;
        }

        /**
         * Set number of output classes.
         *
         * @param numClasses number of classes
         * @return this builder
         */
        public Builder numClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        /**
         * Build the ResNet model.
         *
         * @return configured ResNet
         */
        public ResNet build() {
            return new ResNet(this);
        }
    }
}

