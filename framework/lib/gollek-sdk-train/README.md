# Gollek SDK - Training Pipeline

Complete training pipeline with PyTorch Lightning-like functionality for Java.

## Features

- ✅ **Trainer** - Complete training loop with callbacks
- ✅ **EarlyStopping** - Stop training when metrics stop improving
- ✅ **ModelCheckpoint** - Save best models during training
- ✅ **Learning Rate Schedulers** - StepLR, CosineAnnealingLR
- ✅ **Training Metrics** - Track losses, learning rates, custom metrics
- ✅ **Console Logging** - Configurable training progress output
- ✅ **Gradient Clipping** - Clip by norm (fully implemented)
- ✅ **Mixed Precision** - FP16 training with GradScaler (fully implemented)

## Quick Start

### Basic Training

```java
import tech.kayys.gollek.sdk.train.*;
import tech.kayys.gollek.sdk.autograd.*;
import tech.kayys.gollek.sdk.data.DataLoader;

// Build trainer
Trainer trainer = Trainer.builder()
    .model(myModel)
    .optimizer(new Adam(myModel.parameters(), 0.001))
    .loss((preds, targets) -> preds.mseLoss(targets))
    .epochs(100)
    .gradientClip(1.0)
    .build();

// Run training
trainer.fit(trainLoader, valLoader);
```

### With Callbacks

```java
Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(new Adam(model.parameters(), 0.001))
    .loss(CrossEntropyLoss())
    .callbacks(List.of(
        EarlyStopping.patience(5),
        ModelCheckpoint.at(Path.of("checkpoints/")),
        ConsoleLogger.create()
    ))
    .scheduler(CosineAnnealingLR.builder()
        .maxLr(0.001)
        .minLr(0.00001)
        .tMax(100)
        .build())
    .epochs(100)
    .build();

trainer.fit(trainLoader, valLoader);
```

### Advanced Configuration

```java
Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(new AdamW(model.parameters(), 0.001, 0.01))
    .loss(CrossEntropyLoss())
    .callbacks(List.of(
        EarlyStopping.builder()
            .patience(10)
            .minDelta(0.001)
            .mode(EarlyStopping.Mode.MIN)
            .build(),
        ModelCheckpoint.builder()
            .dirPath(Path.of("checkpoints/"))
            .saveTopK(3)
            .filename("model-{epoch}-{val_loss:.4f}.pt")
            .build(),
        ConsoleLogger.builder()
            .logInterval(10)
            .showMetrics(true)
            .build()
    ))
    .scheduler(StepLR.builder()
        .baseLr(0.001)
        .stepSize(30)
        .gamma(0.1)
        .build())
    .epochs(200)
    .gradientClip(1.0)          // Gradient clipping by norm
    .mixedPrecision(true)       // FP16 training with GradScaler
    .checkpointDir(Path.of("checkpoints/"))
    .build();

trainer.fit(trainLoader, valLoader);

// Access GradScaler for monitoring
GradScaler scaler = trainer.getGradScaler();
if (scaler != null) {
    System.out.println("Current scale: " + scaler.getScale());
    System.out.println("Overflow detected: " + scaler.isOverflowDetected());
}
```

### Mixed Precision Training

Mixed precision training uses FP16 for forward/backward passes while maintaining FP32 master weights, providing:
- 2-3x speedup on GPUs with Tensor Cores
- 50% reduction in memory usage
- Automatic loss scaling to prevent gradient underflow

```java
Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(Adam.create(model.parameters(), 0.001))
    .loss(CrossEntropyLoss())
    .mixedPrecision(true)  // Enable FP16 training
    .gradientClip(1.0)     // Recommended with mixed precision
    .epochs(100)
    .build();

trainer.fit(trainLoader, valLoader);
```

The GradScaler automatically:
1. Scales the loss before backward to prevent gradient underflow
2. Unscales gradients before clipping/optimizer step
3. Detects overflow (inf/NaN) and adjusts scale factor
4. Skips optimizer step on overflow to prevent corruption

## Components

### Trainer

Main training loop that handles:
- Training/validation epochs
- Callback invocation
- Learning rate scheduling
- Gradient clipping
- Mixed precision training

```java
Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(optimizer)
    .loss(lossFn)
    .epochs(100)
    .build();

trainer.fit(trainLoader, valLoader);
```

### Callbacks

Hook into training events:

```java
public interface Callback {
    void onTrainingStart(Trainer trainer);
    void onEpochStart(Trainer trainer, int epoch);
    void onEpochEnd(Trainer trainer, int epoch, double trainLoss);
    void onValidationEnd(Trainer trainer, int epoch, double valLoss);
    void onBatchStart(Trainer trainer, int step);
    void onBatchEnd(Trainer trainer, int step, double loss);
    void onEarlyStopping(Trainer trainer, int epoch);
    void onTrainingError(Trainer trainer, Exception error);
    void onTrainingEnd(Trainer trainer);
}
```

#### EarlyStopping

Stop training when metric stops improving:

```java
Callback earlyStop = EarlyStopping.builder()
    .patience(5)           // Wait 5 epochs
    .minDelta(0.001)       // Minimum improvement
    .mode(Mode.MIN)        // Monitor loss (lower is better)
    .build();
```

#### ModelCheckpoint

Save best models:

```java
Callback checkpoint = ModelCheckpoint.builder()
    .dirPath(Path.of("checkpoints/"))
    .saveTopK(3)           // Keep best 3
    .filename("model-{epoch}-{val_loss:.4f}.pt")
    .build();
```

#### ConsoleLogger

Log training progress:

```java
Callback logger = ConsoleLogger.builder()
    .logInterval(10)       // Log every 10 batches
    .showMetrics(true)     // Show metrics summary
    .build();
```

### Learning Rate Schedulers

#### StepLR

Decay LR by gamma every stepSize epochs:

```java
LRScheduler scheduler = StepLR.builder()
    .baseLr(0.001)
    .stepSize(30)
    .gamma(0.1)
    .build();
```

#### CosineAnnealingLR

Cosine annealing schedule:

```java
LRScheduler scheduler = CosineAnnealingLR.builder()
    .maxLr(0.001)
    .minLr(0.00001)
    .tMax(100)
    .build();
```

### TrainingMetrics

Track and query training metrics:

```java
TrainingMetrics metrics = trainer.getMetrics();

// Get losses
double bestValLoss = metrics.getBestValLoss();
int bestEpoch = metrics.getBestValLossEpoch();
List<Double> trainLosses = metrics.getTrainLosses();

// Get summary
Map<String, Object> summary = metrics.getSummary();
// {epochs=100, best_val_loss=0.123, duration=3600.5s, ...}
```

### GradScaler (Mixed Precision)

Scale gradients for FP16 training:

```java
GradScaler scaler = GradScaler.builder()
    .initScale(65536.0)      // Initial scale factor
    .growthFactor(2.0)       // Multiply scale by this on growth
    .backoffFactor(0.5)      // Multiply scale by this on overflow
    .growthInterval(2000)    // Steps between scale increases
    .minScale(1.0)           // Minimum allowed scale
    .maxScale(1e10)          // Maximum allowed scale
    .build();

// Manual usage (if not using Trainer)
GradTensor loss = model.forward(inputs).mseLoss(targets);
scaler.scale(loss).backward();
scaler.unscaleAndCheck(parameters);
scaler.step(optimizer);
scaler.update();
```

## Maven Dependency

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-sdk-train</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

## API Reference

### Trainer.Builder

| Method | Default | Description |
|--------|---------|-------------|
| `model(Model)` | Required | Neural network to train |
| `optimizer(Optimizer)` | Required | Optimizer instance |
| `loss(LossFunction)` | Required | Loss function |
| `epochs(int)` | 100 | Number of training epochs |
| `gradientClip(double)` | 0.0 (disabled) | Max gradient norm |
| `mixedPrecision(boolean)` | false | Enable FP16 training with GradScaler |
| `checkpointDir(Path)` | null | Checkpoint directory |
| `callback(Callback)` | - | Add callback |
| `scheduler(LRScheduler)` | - | Add LR scheduler |

### GradScaler.Builder

| Method | Default | Description |
|--------|---------|-------------|
| `initScale(double)` | 65536.0 | Initial scale factor |
| `growthFactor(double)` | 2.0 | Scale multiplier on growth |
| `backoffFactor(double)` | 0.5 | Scale multiplier on overflow |
| `growthInterval(int)` | 2000 | Steps between scale increases |
| `minScale(double)` | 1.0 | Minimum allowed scale |
| `maxScale(double)` | 1e10 | Maximum allowed scale |

### EarlyStopping.Builder

| Method | Default | Description |
|--------|---------|-------------|
| `patience(int)` | 5 | Epochs to wait |
| `minDelta(double)` | 0.0 | Min improvement |
| `mode(Mode)` | MIN | MIN for loss, MAX for accuracy |
| `monitor(String)` | "val_loss" | Metric to monitor |

### ModelCheckpoint.Builder

| Method | Default | Description |
|--------|---------|-------------|
| `dirPath(Path)` | `checkpoints/` | Checkpoint directory |
| `filename(String)` | `model-{epoch}-{val_loss}.pt` | Filename pattern |
| `saveTopK(int)` | 1 | Keep best K checkpoints |
| `monitor(String)` | "val_loss" | Metric to monitor |
| `mode(Mode)` | MIN | MIN for loss, MAX for accuracy |

### StepLR.Builder

| Method | Default | Description |
|--------|---------|-------------|
| `baseLr(double)` | 0.001 | Initial learning rate |
| `stepSize(int)` | 30 | Epochs between decays |
| `gamma(double)` | 0.1 | Decay factor |

### CosineAnnealingLR.Builder

| Method | Default | Description |
|--------|---------|-------------|
| `maxLr(double)` | 0.001 | Maximum learning rate |
| `minLr(double)` | 0.0 | Minimum learning rate |
| `tMax(int)` | 100 | Total epochs |

## Examples

### Image Classification

```java
// Model
Module model = new ResNet18(numClasses=10);

// Optimizer
Optimizer optimizer = new AdamW(model.parameters(), 0.001, 0.01);

// Loss
LossFunction loss = CrossEntropyLoss();

// Callbacks
List<Callback> callbacks = List.of(
    EarlyStopping.patience(10),
    ModelCheckpoint.at(Path.of("checkpoints/")),
    ConsoleLogger.create()
);

// Scheduler
LRScheduler scheduler = CosineAnnealingLR.builder()
    .maxLr(0.001)
    .minLr(0.00001)
    .tMax(100)
    .build();

// Trainer
Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(optimizer)
    .loss(loss)
    .callbacks(callbacks)
    .scheduler(scheduler)
    .epochs(100)
    .gradientClip(1.0)
    .build();

// Train
trainer.fit(trainLoader, valLoader);
```

### Custom Callback

```java
public class TensorBoardCallback implements Callback {
    private final TensorBoardWriter writer;

    public TensorBoardCallback(Path logDir) {
        this.writer = new TensorBoardWriter(logDir);
    }

    @Override
    public void onEpochEnd(Trainer trainer, int epoch, double trainLoss) {
        writer.addScalar("train/loss", trainLoss, epoch);
        writer.addScalar("lr", trainer.getOptimizer().getLr(), epoch);
    }

    @Override
    public void onValidationEnd(Trainer trainer, int epoch, double valLoss) {
        writer.addScalar("val/loss", valLoss, epoch);
    }

    @Override
    public void close() {
        writer.close();
    }
}
```

## Roadmap

- [ ] Gradient accumulation
- [ ] Distributed training (DDP)
- [ ] TensorBoard integration
- [ ] Weights & Biases integration
- [ ] Gradient checkpointing
- [ ] Profiling and timing
- [ ] Model pruning during training
- [ ] Knowledge distillation support
- [ ] Automatic mixed precision (AMP) with native FP16 tensors

## License

MIT License - Copyright (c) 2026 Kayys.tech
