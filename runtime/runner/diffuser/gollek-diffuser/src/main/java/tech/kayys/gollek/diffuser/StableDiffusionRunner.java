package tech.kayys.gollek.diffuser;


import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.List;

/**
 * Stable Diffusion loader/runner using JDK 25 Foreign Function & Memory API
 * and the Vector API for SIMD-accelerated post-processing.
 *
 * Targets: JDK 25, --enable-preview
 * Native backend: ONNX Runtime (libonnxruntime.so / .dylib / .dll)
 *
 * Pipeline:
 * prompt → CLIP text encoder → latent noise (PRNG) →
 * U-Net denoising loop → VAE decoder → float[] pixels →
 * VectorizedOps (normalize + clamp via Vector API) → PNG output
 */
public final class StableDiffusionRunner implements AutoCloseable {

    // ── Configuration ────────────────────────────────────────────────────────

    public record Config(
            Path modelDir, // directory with *.onnx files
            int imageWidth, // must be multiple of 8
            int imageHeight,
            int inferenceSteps,
            float guidanceScale,
            long seed) {
        public static Config defaults(Path modelDir) {
            return new Config(modelDir, 512, 512, 20, 7.5f, 42L);
        }
    }

    // ── State ─────────────────────────────────────────────────────────────────

    private final Config config;
    private final Arena arena; // FFM confined arena
    private final OnnxRuntimeBridge ort;
    private final TextEncoder textEncoder;
    private final UNetDenoiser unet;
    private final VaeDecoder vaeDecoder;
    private final DDIMScheduler scheduler;

    // ── Construction ──────────────────────────────────────────────────────────

    public StableDiffusionRunner(Config config) throws Throwable {
        this.config = config;
        this.arena = Arena.ofConfined(); // FFM: lifecycle-bound arena

        NativeLibraryLoader.load(); // dlopen / LoadLibrary

        this.ort = new OnnxRuntimeBridge(arena);
        this.textEncoder = new TextEncoder(ort, config.modelDir().resolve("text_encoder.onnx"), arena);
        this.unet = new UNetDenoiser(ort, config.modelDir().resolve("unet.onnx"), arena);
        this.vaeDecoder = new VaeDecoder(ort, config.modelDir().resolve("vae_decoder.onnx"), arena);
        this.scheduler = new DDIMScheduler(config.inferenceSteps());
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Run full txt2img pipeline.
     *
     * @param prompt    positive prompt
     * @param negPrompt negative prompt (empty string = uncond)
     * @return raw RGBA float[] pixels, shape [H × W × 4], range [0,1]
     */
    public float[] generate(String prompt, String negPrompt) throws Throwable {
        // 1. Encode text (FFM call → ONNX Runtime)
        MemorySegment condEmbeds = textEncoder.encode(prompt, config.imageWidth());
        MemorySegment uncondEmbeds = textEncoder.encode(negPrompt, config.imageWidth());

        // 2. Initial latents (random Gaussian, seeded)
        int latentH = config.imageHeight() / 8;
        int latentW = config.imageWidth() / 8;
        MemorySegment latents = initLatents(latentH, latentW);

        // 3. Denoising loop
        List<Float> timesteps = scheduler.timesteps();
        for (int i = 0; i < timesteps.size(); i++) {
            float t = timesteps.get(i);

            // classifier-free guidance: run U-Net twice, merge
            MemorySegment noisePredCond = unet.predict(latents, condEmbeds, t);
            MemorySegment noisePredUncond = unet.predict(latents, uncondEmbeds, t);

            // guidance: noise = uncond + scale * (cond - uncond) [Vector API]
            MemorySegment guidedNoise = VectorizedOps.classifierFreeGuidance(
                    noisePredUncond, noisePredCond, config.guidanceScale(), arena);

            latents = scheduler.step(latents, guidedNoise, t, arena);

            System.out.printf("  step %2d/%d  t=%.1f%n", i + 1, timesteps.size(), t);
        }

        // 4. Decode latents → pixel space (FFM call → VAE)
        MemorySegment decoded = vaeDecoder.decode(latents);

        // 5. Post-process: normalize + clamp via Vector API
        return VectorizedOps.postProcess(decoded, config.imageWidth(), config.imageHeight());
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /** Allocate & fill latent tensor with seeded Gaussian noise (off-heap). */
    private MemorySegment initLatents(int h, int w) {
        int size = 1 * 4 * h * w; // batch=1, channels=4
        MemorySegment seg = arena.allocate(size * Float.BYTES, 8);

        java.util.Random rng = new java.util.Random(config.seed());
        for (int i = 0; i < size; i++) {
            seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) rng.nextGaussian());
        }
        return seg;
    }

    // ── AutoCloseable ─────────────────────────────────────────────────────────

    @Override
    public void close() {
        arena.close(); // frees all FFM off-heap allocations
        ort.close();
    }
}
