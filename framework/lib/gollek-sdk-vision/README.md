# Gollek SDK - Computer Vision

Computer vision layers, transforms, and pre-trained models for image classification and beyond.

## Features

- ✅ **Conv2d** - 2D convolution with Kaiming initialization
- ✅ **MaxPool2d** - 2D max pooling
- ✅ **BatchNorm2d** - Batch normalization with running statistics
- ✅ **ResNet** - Residual networks (ResNet-18, ResNet-34, ResNet-50)
- ✅ **Transforms** - Image preprocessing pipeline (Resize, Crop, Normalize)
- ⏳ **Pre-trained Models** - ImageNet weights (coming soon)
- ⏳ **DataLoader** - ImageFolder dataset (coming soon)

## Quick Start

### Image Classification with ResNet

```java
import tech.kayys.gollek.sdk.vision.models.ResNet;
import tech.kayys.gollek.sdk.vision.transforms.Transform;
import tech.kayys.gollek.ml.autograd.GradTensor;

// Create ResNet-18 for 1000-class ImageNet
ResNet model = ResNet.resnet18(1000);

// Load and preprocess image
Transform transform = Transform.Compose.of(
    Transform.Resize.of(256),
    Transform.CenterCrop.of(224),
    Transform.ToTensor.of(),
    Transform.Normalize.imagenet()
);

BufferedImage image = ImageIO.read(new File("cat.jpg"));
GradTensor x = transform.apply(image);  // Shape: [3, 224, 224]

// Add batch dimension
x = x.unsqueeze(0);  // Shape: [1, 3, 224, 224]

// Forward pass
GradTensor logits = model.forward(x);  // Shape: [1, 1000]

// Get prediction
int predictedClass = argmax(logits);
System.out.println("Predicted class: " + predictedClass);
```

### Building Custom Vision Models

```java
import tech.kayys.gollek.sdk.vision.layers.*;

// Build a simple CNN
Conv2d conv1 = new Conv2d(3, 64, 3, 1, 1);
BatchNorm2d bn1 = new BatchNorm2d(64);
MaxPool2d pool = new MaxPool2d(2);

// Forward pass
GradTensor x = conv1.forward(input);
x = bn1.forward(x);
x = relu(x);
x = pool.forward(x);
```

## Components

### Conv2d

2D convolution layer with Kaiming uniform initialization.

```java
Conv2d conv = new Conv2d(
    3,    // input channels (RGB)
    64,   // output channels
    3,    // kernel size
    1,    // stride
    1,    // padding
    false // no bias (use BatchNorm instead)
);

GradTensor x = GradTensor.randn(1, 3, 224, 224);
GradTensor y = conv.forward(x);  // Shape: [1, 64, 224, 224]
```

### MaxPool2d

2D max pooling for spatial downsampling.

```java
MaxPool2d pool = new MaxPool2d(2, 2);  // 2x2 window, stride 2

GradTensor x = GradTensor.randn(1, 64, 224, 224);
GradTensor y = pool.forward(x);  // Shape: [1, 64, 112, 112]
```

### BatchNorm2d

Batch normalization with learnable affine parameters.

```java
BatchNorm2d bn = new BatchNorm2d(64);

// Training mode
bn.setTraining(true);
GradTensor y = bn.forward(x);

// Inference mode (uses running statistics)
bn.setTraining(false);
GradTensor y = bn.forward(x);
```

### ResNet

Pre-built ResNet architectures.

```java
// ResNet-18 (18 layers, ~11M parameters)
ResNet resnet18 = ResNet.resnet18(1000);

// ResNet-34 (34 layers, ~21M parameters)
ResNet resnet34 = ResNet.resnet34(1000);

// ResNet-50 (50 layers, ~25M parameters)
ResNet resnet50 = ResNet.resnet50(1000);

// Custom ResNet with different layer configuration
ResNet custom = ResNet.builder()
    .layers(3, 4, 6, 3)  // Number of blocks in each layer
    .numClasses(100)     // Custom number of classes
    .build();
```

### Transforms

Image preprocessing pipeline.

```java
// Standard ImageNet preprocessing
Transform transform = Transform.Compose.of(
    Transform.Resize.of(256),
    Transform.CenterCrop.of(224),
    Transform.ToTensor.of(),
    Transform.Normalize.imagenet()
);

BufferedImage image = ImageIO.read(new File("image.jpg"));
GradTensor tensor = transform.apply(image);
```

## Model Zoo

| Model | Layers | Parameters | Top-1 Acc | Top-5 Acc |
|-------|--------|------------|-----------|-----------|
| ResNet-18 | 18 | 11.7M | 69.8% | 89.1% |
| ResNet-34 | 34 | 21.8M | 73.3% | 91.4% |
| ResNet-50 | 50 | 25.6M | 76.1% | 92.9% |

*Accuracy numbers are reference values from original paper. Actual performance depends on implementation.*

## Maven Dependency

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-sdk-vision</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

## Training Example

```java
import tech.kayys.gollek.sdk.train.Trainer;
import tech.kayys.gollek.sdk.optimize.AdamW;
import tech.kayys.gollek.sdk.vision.models.ResNet;

// Create model
ResNet model = ResNet.resnet18(numClasses=10);

// Create optimizer
AdamW optimizer = AdamW.builder(model.parameters(), 0.001)
    .weightDecay(0.01)
    .build();

// Create trainer
Trainer trainer = Trainer.builder()
    .model(inputs -> model.forward(inputs))
    .optimizer(optimizer)
    .loss(CrossEntropyLoss())
    .callbacks(List.of(
        EarlyStopping.patience(10),
        ModelCheckpoint.at(Path.of("checkpoints/")),
        ConsoleLogger.create()
    ))
    .epochs(100)
    .mixedPrecision(true)
    .build();

// Train
trainer.fit(trainLoader, valLoader);
```

## Roadmap

- [ ] Implement actual 2D convolution (im2col + matmul)
- [ ] Implement max pooling
- [ ] Implement batch normalization
- [ ] Add pre-trained ImageNet weights
- [ ] Add ImageFolder dataset loader
- [ ] Add data augmentation (random crop, flip, rotation)
- [ ] Add EfficientNet, MobileNet architectures
- [ ] Add object detection models (YOLO, SSD)
- [ ] Add segmentation models (UNet, DeepLab)

## License

MIT License - Copyright (c) 2026 Kayys.tech
