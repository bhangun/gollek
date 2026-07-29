package tech.kayys.gollek.gguf.runner;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import tech.kayys.gollek.gguf.runtime.GgufBudget;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProbe;
import tech.kayys.gollek.gguf.runtime.GgufTensorOps;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;
import tech.kayys.gollek.gguf.model.alkhawarizm.LlamaModel;
import tech.kayys.gollek.gguf.model.alkhawarizm.SigLIPVisionTransformer;
import tech.kayys.gollek.gguf.model.alkhawarizm.VisionProjector;
import tech.kayys.gollek.gguf.runner.AljabrWeightAdapter;
import tech.kayys.gollek.gguf.loader.model.ModelConfig;

/**
 * Java-native GGUF backend scaffold backed by the active GGUF tensor
 * primitives.
 */
public class JavaNativeGgufBackend implements GgufBackend {
    private final GGUFModel model;
    private final GgufRuntimeProfile profile;
    private final GgufRuntimeProbe.PreparedMatrixCacheDecision preparedCacheDecision;

    public JavaNativeGgufBackend(Path modelPath) throws Exception {
        long startNanos = System.nanoTime();
        this.model = GGUFLoader.loadModel(modelPath);
        long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
        this.profile = GgufRuntimeProfile.fromModel(model, Files.size(modelPath), loadMillis);
        this.preparedCacheDecision = prepareDecoderMatrixCaches(model);
    }

    JavaNativeGgufBackend(GGUFModel model, long modelBytes, long loadMillis) {
        this.model = model;
        this.profile = GgufRuntimeProfile.fromModel(model, modelBytes, loadMillis);
        this.preparedCacheDecision = prepareDecoderMatrixCaches(model);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        try {
            // 1. Load weights into Aljabr Tensors
            AljabrWeightAdapter weights = new AljabrWeightAdapter(model.file());
            ModelConfig config = new ModelConfig(model.file());

            // 2. Initialize Aljabr-NN Modules
            LlamaModel textModel = new LlamaModel(config, weights);

            boolean isMultimodal = config.contains("vision_config");
            SigLIPVisionTransformer visionModel = null;
            VisionProjector projector = null;

            if (isMultimodal) {
                visionModel = new SigLIPVisionTransformer(config, weights);
                projector = new VisionProjector(weights);
            }

            // 3. Process inputs (Scaffold for generation loop)
            List<String> imagePaths = (List<String>) request.getParameter("vision_input_paths").orElse(null);
            Tensor visionEmbeds = null;
            if (imagePaths != null && !imagePaths.isEmpty() && visionModel != null) {
                // Load first image
                String path = imagePaths.get(0);
                java.awt.image.BufferedImage img = javax.imageio.ImageIO.read(new java.io.File(path));
                if (img != null) {
                    int targetWidth = 384; // config.getInt("vision_config.image_size", 384)
                    int targetHeight = targetWidth; // usually square
                    java.awt.Image tmp = img.getScaledInstance(targetWidth, targetHeight, java.awt.Image.SCALE_SMOOTH);
                    java.awt.image.BufferedImage resized = new java.awt.image.BufferedImage(targetWidth, targetHeight,
                            java.awt.image.BufferedImage.TYPE_INT_RGB);
                    java.awt.Graphics2D g2d = resized.createGraphics();
                    g2d.drawImage(tmp, 0, 0, null);
                    g2d.dispose();

                    float[] mean = { 0.5f, 0.5f, 0.5f };
                    float[] std = { 0.5f, 0.5f, 0.5f };
                    float[] tensorData = new float[3 * targetHeight * targetWidth];
                    for (int y = 0; y < targetHeight; y++) {
                        for (int x = 0; x < targetWidth; x++) {
                            int rgb = resized.getRGB(x, y);
                            float r = ((rgb >> 16) & 0xFF) / 255.0f;
                            float g = ((rgb >> 8) & 0xFF) / 255.0f;
                            float b = (rgb & 0xFF) / 255.0f;
                            tensorData[0 * targetHeight * targetWidth + y * targetWidth + x] = (r - mean[0]) / std[0];
                            tensorData[1 * targetHeight * targetWidth + y * targetWidth + x] = (g - mean[1]) / std[1];
                            tensorData[2 * targetHeight * targetWidth + y * targetWidth + x] = (b - mean[2]) / std[2];
                        }
                    }
                    Tensor imageTensor = TensorFactory.of(tensorData, 1, 3, targetHeight, targetWidth);
                    Tensor visionOut = visionModel.forward(imageTensor);
                    visionEmbeds = projector.forward(visionOut);
                }
            }

            // Tokenize prompt and run prefill/decode loop using textModel.forward()
            System.out.println("Starting Java-native generation loop...");

            // Dummy prompt (since tokenizer is pending)
            Tensor promptIds = TensorFactory.of(new float[] { 1, 2, 3 }, 1, 3);

            for (int i = 0; i < 5; i++) {
                long startMs = System.currentTimeMillis();

                // Pass visionEmbeds on the first step (prefill), then null for decode steps
                Tensor logits = textModel.forward(promptIds, i == 0 ? visionEmbeds : null);

                long elapsed = System.currentTimeMillis() - startMs;
                System.out.println("Step " + (i + 1) + ": textModel.forward() took " + elapsed + "ms. Logits shape: "
                        + logits.shape());

                // Dummy decode (just appending a token)
                promptIds = TensorFactory.of(new float[] { 4 }, 1, 1);
            }

            return (RunnerResult<T>) RunnerResult.success("Aljabr native engine successfully initialized graph with "
                    + textModel.parameterCount() + " parameters and executed generation loop.");
        } catch (Exception e) {
            return RunnerResult.failed("Aljabr native engine failed: " + e.getMessage());
        }
    }

    @Override
    public void close() {
        GgufTensorOps.clearPreparedMatrixCaches(model);
        model.close();
    }

    GgufTensorOps.PreparedMatrixCachePlan preparedCachePlan() {
        return preparedCacheDecision.plan();
    }

    GgufTensorOps.PreparedMatrixCacheStats preparedCacheStats() {
        return preparedCacheDecision.stats();
    }

    String preparedCacheSummary() {
        return preparedCacheDecision.compactSummary();
    }

    private static GgufRuntimeProbe.PreparedMatrixCacheDecision prepareDecoderMatrixCaches(GGUFModel model) {
        int explicitMinRows = Math.max(0, Integer.getInteger("gollek.gguf.java_native.prepare_min_rows", 0));
        if (explicitMinRows > 0) {
            return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                    model,
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(
                            model,
                            explicitMinRows,
                            false,
                            1,
                            0L));
        }

        int autoMinRows = Math.max(1, Integer.getInteger("gollek.gguf.java_native.auto_prepare_min_rows", 32));
        long budgetBytes = autoPrepareBudgetBytes();
        return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                model,
                GgufRuntimeProbe.selectDecoderPreparedMatrixCache(
                        model,
                        0,
                        Boolean.parseBoolean(System.getProperty("gollek.gguf.java_native.auto_prepare", "true")),
                        autoMinRows,
                        budgetBytes));
    }

    private static long autoPrepareBudgetBytes() {
        return GgufBudget.byteSizeProperty(
                "gollek.gguf.java_native.auto_prepare_budget_bytes",
                GgufBudget.defaultAutoPrepareBytes());
    }

}
