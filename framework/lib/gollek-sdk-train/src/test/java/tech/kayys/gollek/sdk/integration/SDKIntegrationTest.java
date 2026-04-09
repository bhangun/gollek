package tech.kayys.gollek.sdk.integration;

import org.junit.jupiter.api.*;
import tech.kayys.gollek.sdk.core.*;
import tech.kayys.gollek.sdk.optimize.*;
import tech.kayys.gollek.sdk.vision.models.ResNet;
import tech.kayys.gollek.sdk.vision.layers.*;
import tech.kayys.gollek.sdk.augment.*;
import tech.kayys.gollek.sdk.export.*;
import tech.kayys.gollek.sdk.export.Benchmark.BenchmarkResult;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for all SDK modules working together.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SDKIntegrationTest {

    @Test
    @Order(1)
    @DisplayName("Test 1: Tensor API with device placement")
    void testTensorAPI() {
        // Create tensors
        Tensor x = Tensor.randn(2, 3, 224, 224);
        Tensor y = Tensor.zeros(2, 1000);

        // Device placement
        Tensor x_cpu = x.to(Device.CPU);
        assertEquals(Device.CPU, x_cpu.device());

        // Operations
        Tensor z = x.add(y.unsqueeze(2).unsqueeze(3));
        assertNotNull(z);

        // Reshaping
        Tensor flat = x.flatten();
        assertEquals(1, flat.ndim());

        // Activations
        Tensor relu = x.relu();
        Tensor sigmoid = x.sigmoid();
        Tensor softmax = x.softmax();

        assertNotNull(relu);
        assertNotNull(sigmoid);
        assertNotNull(softmax);
    }

    @Test
    @Order(2)
    @DisplayName("Test 2: Vision layers forward pass")
    void testVisionLayers() {
        // Conv2d
        Conv2d conv = new Conv2d(3, 64, 3, 1, 1);
        GradTensor x = GradTensor.randn(1, 3, 224, 224);
        GradTensor y = conv.forward(x);
        assertEquals(4, y.shape().length);
        assertEquals(64, y.shape()[1]);

        // BatchNorm2d
        BatchNorm2d bn = new BatchNorm2d(64);
        GradTensor z = bn.forward(y);
        assertEquals(y.shape().length, z.shape().length);

        // MaxPool2d
        MaxPool2d pool = new MaxPool2d(2);  // kernel=2, stride=2, padding=0
        GradTensor p = pool.forward(z);
        assertEquals(112, p.shape()[2]);
        assertEquals(112, p.shape()[3]);

        // Linear
        Linear fc = new Linear(512, 10);
        GradTensor input = GradTensor.randn(2, 512);
        GradTensor output = fc.forward(input);
        assertEquals(2, output.shape()[0]);
        assertEquals(10, output.shape()[1]);
    }

    @Test
    @Order(3)
    @DisplayName("Test 3: ResNet model creation and forward pass")
    void testResNet() {
        // Create different ResNet variants
        ResNet resnet18 = ResNet.resnet18(1000);
        ResNet resnet34 = ResNet.resnet34(1000);
        ResNet resnet50 = ResNet.resnet50(1000);

        assertNotNull(resnet18);
        assertNotNull(resnet34);
        assertNotNull(resnet50);

        // Forward pass
        GradTensor input = GradTensor.randn(1, 3, 224, 224);
        GradTensor output = resnet18.forward(input);

        assertEquals(2, output.shape().length);
        assertEquals(1, output.shape()[0]);
        assertEquals(1000, output.shape()[1]);
    }

    @Test
    @Order(4)
    @DisplayName("Test 4: Data augmentation pipeline")
    void testAugmentation() {
        // Create pipeline
        AugmentationPipeline pipeline = new AugmentationPipeline(
            Augmentation.RandomHorizontalFlip.create(),
            Augmentation.RandomCrop.of(224),
            Augmentation.ColorJitter.of(0.2, 0.2, 0.2)
        );

        assertEquals(3, pipeline.size());

        // Apply to sample data
        GradTensor input = GradTensor.randn(3, 256, 256);
        GradTensor output = pipeline.apply(input);

        assertNotNull(output);
    }

    @Test
    @Order(5)
    @DisplayName("Test 5: Optimizer creation")
    void testOptimizers() {
        // Create sample parameters
        GradTensor param = GradTensor.randn(10, 10);
        param.requiresGrad(true);
        List<GradTensor> params = List.of(param);

        // Adam
        Optimizer adam = Adam.builder(params, 0.001)
            .betas(0.9, 0.999)
            .weightDecay(0.01)
            .build();
        assertNotNull(adam);
        assertEquals(0.001, adam.getLr());

        // AdamW
        Optimizer adamw = AdamW.builder(params, 0.001)
            .weightDecay(0.01)
            .build();
        assertNotNull(adamw);

        // SGD
        Optimizer sgd = SGD.builder(params, 0.01)
            .momentum(0.9)
            .nesterov(true)
            .build();
        assertNotNull(sgd);

        // RMSprop
        Optimizer rmsprop = RMSprop.builder(params, 0.01)
            .alpha(0.99)
            .build();
        assertNotNull(rmsprop);
    }

    @Test
    @Order(6)
    @DisplayName("Test 6: Model export")
    void testModelExport() throws Exception {
        // Create model
        ResNet model = ResNet.resnet18(10);

        // Create exporter
        ModelExporter exporter = ModelExporter.builder()
            .model(model)
            .inputShape(1, 3, 224, 224)
            .build();

        assertNotNull(exporter);

        // Create temp directory
        Path tempDir = Path.of("target/test-exports");
        java.nio.file.Files.createDirectories(tempDir);

        // Export to different formats
        exporter.toONNX(tempDir.resolve("model.onnx"));
        exporter.toGGUF(tempDir.resolve("model.gguf"), ModelExporter.Quantization.INT4);
        exporter.toLiteRT(tempDir.resolve("model.litert"));

        // Verify files created
        assertTrue(java.nio.file.Files.exists(tempDir.resolve("model.onnx")));
        assertTrue(java.nio.file.Files.exists(tempDir.resolve("model.gguf")));
        assertTrue(java.nio.file.Files.exists(tempDir.resolve("model.litert")));
    }

    @Test
    @Order(7)
    @DisplayName("Test 7: Benchmark")
    void testBenchmark() {
        // Create model
        ResNet model = ResNet.resnet18(10);

        // Run benchmark
        Benchmark benchmark = new Benchmark(model);
        BenchmarkResult result = benchmark.run(new long[]{1, 3, 224, 224}, 10, 5);

        assertNotNull(result);
        assertTrue(result.avgLatencyMs() > 0);
        assertTrue(result.throughput() > 0);
        assertEquals(10, result.iterations());
    }

    @Test
    @Order(8)
    @DisplayName("Test 8: Memory management")
    void testMemoryManager() {
        MemoryManager mem = MemoryManager.getInstance();
        mem.reset();

        // Track allocations
        mem.allocate(1024 * 1024);  // 1 MB
        assertEquals(1024 * 1024, mem.getAllocatedBytes());
        assertEquals(1024 * 1024, mem.getPeakBytes());

        // Free memory
        mem.free(512 * 1024);  // 512 KB
        assertEquals(512 * 1024, mem.getAllocatedBytes());

        // Check stats
        String stats = mem.getStats();
        assertNotNull(stats);
        assertTrue(stats.contains("allocated"));
    }

    @Test
    @Order(9)
    @DisplayName("Test 9: NoGrad context")
    void testNoGrad() {
        // Initially enabled
        assertTrue(NoGrad.isEnabled());

        // Disable with context
        try (NoGrad ctx = new NoGrad()) {
            assertFalse(NoGrad.isEnabled());
        }

        // Re-enabled after context
        assertTrue(NoGrad.isEnabled());
    }

    @Test
    @Order(10)
    @DisplayName("Test 10: End-to-end workflow")
    void testEndToEndWorkflow() throws Exception {
        // 1. Create model
        ResNet model = ResNet.resnet18(10);

        // 2. Create augmentation pipeline
        AugmentationPipeline augment = new AugmentationPipeline(
            Augmentation.RandomHorizontalFlip.create(),
            Augmentation.RandomCrop.of(224)
        );

        // 3. Create optimizer
        GradTensor param = GradTensor.randn(10, 10);
        param.requiresGrad(true);
        Optimizer optimizer = Adam.create(List.of(param), 0.001);

        // 4. Create exporter
        ModelExporter exporter = ModelExporter.builder()
            .model(model)
            .inputShape(1, 3, 224, 224)
            .build();

        // 5. Run benchmark
        Benchmark benchmark = new Benchmark(model);
        BenchmarkResult result = benchmark.run(new long[]{1, 3, 224, 224});

        // Verify all components work together
        assertNotNull(model);
        assertNotNull(augment);
        assertNotNull(optimizer);
        assertNotNull(exporter);
        assertNotNull(result);
    }
}
