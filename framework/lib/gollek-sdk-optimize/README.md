# Gollek SDK - Optimizer Suite

Complete optimizer suite for neural network training with Adam, AdamW, SGD, and RMSprop.

## Features

- ✅ **Adam** - Adaptive Moment Estimation with AMSGrad variant
- ✅ **AdamW** - Adam with decoupled weight decay
- ✅ **SGD** - Stochastic Gradient Descent with momentum and Nesterov
- ✅ **RMSprop** - Root Mean Square Propagation
- ✅ **Gradient Clipping** - By norm and by value
- ✅ **Weight Decay** - L2 regularization (Adam/AdamW style)
- ✅ **Builder Pattern** - Fluent API for configuration

## Quick Start

### Adam Optimizer

```java
import tech.kayys.gollek.sdk.optimize.Adam;
import tech.kayys.gollek.ml.autograd.GradTensor;

List<GradTensor> parameters = model.parameters();

Optimizer optimizer = Adam.builder(parameters, 0.001)
    .betas(0.9, 0.999)
    .eps(1e-8)
    .weightDecay(0.01)
    .amsgrad(false)
    .build();

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    for (Batch batch : trainLoader) {
        GradTensor loss = model.forward(batch.inputs)
            .mseLoss(batch.targets);
        
        loss.backward();
        optimizer.step();
        optimizer.zeroGrad();
    }
}
```

### AdamW (Recommended for Transformers)

```java
import tech.kayys.gollek.sdk.optimize.AdamW;

Optimizer optimizer = AdamW.builder(parameters, 0.001)
    .betas(0.9, 0.999)
    .weightDecay(0.01)  // Decoupled weight decay
    .build();
```

### SGD with Momentum

```java
import tech.kayys.gollek.sdk.optimize.SGD;

Optimizer optimizer = SGD.builder(parameters, 0.01)
    .momentum(0.9)
    .nesterov(true)       // Nesterov accelerated gradient
    .weightDecay(0.0001)
    .build();
```

### RMSprop

```java
import tech.kayys.gollek.sdk.optimize.RMSprop;

Optimizer optimizer = RMSprop.builder(parameters, 0.01)
    .alpha(0.99)
    .eps(1e-8)
    .momentum(0.0)
    .build();
```

## Gradient Clipping

Prevent exploding gradients:

```java
// Clip by global norm
optimizer.clipGradNorm(1.0);

// Clip by value
optimizer.clipGradValue(-5.0, 5.0);
```

## Optimizer Comparison

| Optimizer | Best For | Speed | Memory | Generalization |
|-----------|----------|-------|--------|----------------|
| **Adam** | Quick prototyping | Fast | Medium | Good |
| **AdamW** | Transformers, LLMs | Fast | Medium | **Best** |
| **SGD** | CNNs, final tuning | Medium | Low | **Best** |
| **RMSprop** | RNNs, RL | Fast | Medium | Good |

## API Reference

### Adam

| Method | Default | Description |
|--------|---------|-------------|
| `betas(beta1, beta2)` | 0.9, 0.999 | Exponential decay rates |
| `eps(eps)` | 1e-8 | Numerical stability |
| `weightDecay(wd)` | 0.0 | L2 regularization |
| `amsgrad(enabled)` | false | Use AMSGrad variant |

### AdamW

| Method | Default | Description |
|--------|---------|-------------|
| `betas(beta1, beta2)` | 0.9, 0.999 | Exponential decay rates |
| `eps(eps)` | 1e-8 | Numerical stability |
| `weightDecay(wd)` | 0.01 | Decoupled weight decay |

### SGD

| Method | Default | Description |
|--------|---------|-------------|
| `momentum(m)` | 0.0 | Momentum factor |
| `nesterov(enabled)` | false | Nesterov accelerated gradient |
| `weightDecay(wd)` | 0.0 | L2 regularization |

### RMSprop

| Method | Default | Description |
|--------|---------|-------------|
| `alpha(alpha)` | 0.99 | Smoothing constant |
| `eps(eps)` | 1e-8 | Numerical stability |
| `momentum(m)` | 0.0 | Momentum factor |
| `weightDecay(wd)` | 0.0 | L2 regularization |

## Maven Dependency

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-sdk-optimize</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

## Integration with Trainer

```java
import tech.kayys.gollek.sdk.train.Trainer;
import tech.kayys.gollek.sdk.optimize.AdamW;

Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(AdamW.builder(model.parameters(), 0.001)
        .weightDecay(0.01)
        .build())
    .loss((preds, targets) -> preds.mse(targets))
    .epochs(100)
    .build();

trainer.fit(trainLoader, valLoader);
```

## License

MIT License - Copyright (c) 2026 Kayys.tech
