package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nlp.Pipeline;
import tech.kayys.gollek.ml.nlp.PipelineConfig;
import tech.kayys.gollek.ml.nlp.PipelineFactory;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.sdk.hub.ModelHub;
import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.sdk.api.GollekSdkBuilder;
import tech.kayys.gollek.sdk.multimodal.*;
import tech.kayys.gollek.runtime.tensor.Device;

import java.util.Map;

/**
 * Main entry point for the Gollek ML framework.
 *
 * <p>
 * Provides static factory methods for creating tensors, building neural
 * networks, running NLP pipelines, performing multimodal inference,
 * and querying device availability — mirroring the top-level
 * {@code torch} namespace in PyTorch while extending it with
 * multimodal AI capabilities.
 *
 * <h3>Quick Start — Tensors &amp; Neural Networks</h3>
 * 
 * <pre>{@code
 * var x = Gollek.tensor(new float[]{1, 2, 3, 4}, 2, 2);
 * var w = Gollek.randn(2, 3).requiresGrad(true);
 * var y = x.matmul(w).relu().sum();
 * y.backward();
 *
 * var model = new Sequential(
 *     new Linear(784, 256),
 *     new ReLU(),
 *     new Linear(256, 10)
 * );
 * }</pre>
 *
 * <h3>Quick Start — NLP Pipelines</h3>
 * 
 * <pre>{@code
 * var gen = Gollek.pipeline("text-generation", "Qwen/Qwen2.5-0.5B");
 * System.out.println(gen.process("Tell me about Java"));
 * }</pre>
 *
 * <h3>Quick Start — Multimodal AI</h3>
 * 
 * <pre>{@code
 * // Vision: describe an image
 * var result = Gollek.vision("gemini-2.0-flash")
 *         .image(Path.of("photo.jpg"))
 *         .prompt("What's in this image?")
 *         .generate();
 * }</pre>
 */
public final class Gollek {

    /** Framework version. */
    public static final String VERSION = "0.1.2";

    private static volatile GollekSdk cachedSdk;

    private Gollek() {}

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Creation
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor tensor(float[] data, long... shape) {
        return GradTensor.of(data, shape);
    }

    public static GradTensor tensor(float... data) {
        return GradTensor.of(data);
    }

    public static GradTensor zeros(long... shape) {
        return GradTensor.zeros(shape);
    }

    public static GradTensor ones(long... shape) {
        return GradTensor.ones(shape);
    }

    public static GradTensor randn(long... shape) {
        return GradTensor.randn(shape);
    }

    public static GradTensor rand(long... shape) {
        return GradTensor.rand(shape);
    }

    public static GradTensor arange(float start, float end, float step) {
        return GradTensor.arange(start, end, step);
    }

    public static GradTensor arange(int end) {
        return GradTensor.arange(0, end, 1);
    }

    public static GradTensor scalar(float value) {
        return GradTensor.scalar(value);
    }

    public static GradTensor eye(int n) {
        return GradTensor.eye(n);
    }

    public static GradTensor full(float value, long... shape) {
        return GradTensor.full(value, shape);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Gradient Control
    // ══════════════════════════════════════════════════════════════════════

    public static NoGrad noGrad() {
        return NoGrad.enter();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Pipeline Factory
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Creates an NLP pipeline for the given task and model.
     *
     * @param <I>     input type
     * @param <O>     output type
     * @param task    task name (e.g., "text-generation", "embedding")
     * @param modelId model identifier
     * @return configured pipeline
     */
    public static <I, O> Pipeline<I, O> pipeline(String task, String modelId) {
        return PipelineFactory.create(task, modelId);
    }

    /**
     * Creates an NLP pipeline from a configuration object.
     */
    public static <I, O> Pipeline<I, O> pipeline(PipelineConfig config) {
        return PipelineFactory.create(config);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Multimodal Builders
    // ══════════════════════════════════════════════════════════════════════

    public static VisionBuilder vision(String model) {
        return new VisionBuilder(model, resolveDefaultSdk());
    }

    public static AudioBuilder audio(String model) {
        return new AudioBuilder(model, resolveDefaultSdk());
    }

    public static VideoBuilder video(String model) {
        return new VideoBuilder(model, resolveDefaultSdk());
    }

    public static MultimodalBuilder multimodal(String model) {
        return new MultimodalBuilder(model, resolveDefaultSdk());
    }

    // ══════════════════════════════════════════════════════════════════════
    // SDK Builder
    // ══════════════════════════════════════════════════════════════════════

    public static GollekSdkBuilder sdk() {
        return GollekSdk.builder();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Hub
    // ══════════════════════════════════════════════════════════════════════

    public static Map<String, GradTensor> loadWeights(String modelId) throws java.io.IOException {
        return ModelHub.loadWeights(modelId);
    }

    public static void loadInto(NNModule module, String modelId) throws java.io.IOException {
        ModelHub.loadInto(module, modelId);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static boolean isCudaAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        return (os.contains("linux") || os.contains("windows")) && System.getenv("CUDA_PATH") != null;
    }

    public static boolean isMetalAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        return os.contains("mac") && arch.equals("aarch64");
    }

    public static Device defaultDevice() {
        if (isCudaAvailable()) return Device.CUDA;
        if (isMetalAvailable()) return Device.METAL;
        return Device.CPU;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Info
    // ══════════════════════════════════════════════════════════════════════

    public static void printInfo() {
        System.out.println("╔══════════════════════════════════════════╗");
        System.out.println("║          Gollek ML Framework             ║");
        System.out.println("║          Version " + VERSION + "                    ║");
        System.out.println("╠══════════════════════════════════════════╣");
        System.out.println("║  Device: " + pad(defaultDevice().toString(), 31) + "║");
        System.out.println("║  CUDA:   " + pad(String.valueOf(isCudaAvailable()), 31) + "║");
        System.out.println("║  Metal:  " + pad(String.valueOf(isMetalAvailable()), 31) + "║");
        System.out.println("║  Java:   " + pad(System.getProperty("java.version"), 31) + "║");
        try {
            var tasks = PipelineFactory.availableTasks();
            System.out.println("║  Tasks:  " + pad(String.join(", ", tasks), 31) + "║");
        } catch (Exception ignored) {
            System.out.println("║  Tasks:  " + pad("nlp-not-found", 31) + "║");
        }
        System.out.println("╚══════════════════════════════════════════╝");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Internal
    // ══════════════════════════════════════════════════════════════════════

    private static GollekSdk resolveDefaultSdk() {
        GollekSdk result = cachedSdk;
        if (result == null) {
            synchronized (Gollek.class) {
                result = cachedSdk;
                if (result == null) {
                    try {
                        cachedSdk = result = GollekSdk.builder().build();
                    } catch (Exception e) {
                        throw new IllegalStateException(
                                "No GollekSdkProvider found. Add a provider to classpath.", e);
                    }
                }
            }
        }
        return result;
    }

    private static String pad(String s, int len) {
        if (s == null) s = "null";
        if (s.length() >= len) return s.substring(0, len);
        return s + " ".repeat(len - s.length());
    }
}
