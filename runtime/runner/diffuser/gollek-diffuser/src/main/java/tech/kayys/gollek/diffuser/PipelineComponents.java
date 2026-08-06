package tech.kayys.gollek.diffuser;


import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

// ═══════════════════════════════════════════════════════════════════════════════
// UNetDenoiser
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * U-Net noise predictor.
 *
 * Model: unet.onnx
 * Inputs:
 * sample [1, 4, H/8, W/8] float32 – noisy latents
 * timestep [1] float32 – current timestep
 * encoder_hidden_states [1, 77, 768] float32 – CLIP embeddings
 * Output:
 * out_sample [1, 4, H/8, W/8] float32 – predicted noise
 */
class UNetDenoiser {

    private final OnnxRuntimeBridge ort;
    private final MemorySegment session;
    private final Arena arena;

    UNetDenoiser(OnnxRuntimeBridge ort, Path modelPath, Arena arena) throws Throwable {
        this.ort = ort;
        this.arena = arena;
        this.session = ort.createSession(modelPath);
        System.out.println("[UNetDenoiser] Loaded: " + modelPath);
    }

    /**
     * Single U-Net forward pass.
     *
     * @param latents    noisy latent [1, 4, H/8, W/8]
     * @param embeddings CLIP embeddings [1, 77, 768]
     * @param timestep   current diffusion timestep scalar
     * @return predicted noise [1, 4, H/8, W/8]
     */
    MemorySegment predict(MemorySegment latents, MemorySegment embeddings, float timestep)
            throws Throwable {

        long latentFloats = latents.byteSize() / Float.BYTES;
        int latentH = deriveLatentDim(latentFloats, 0); // H/8
        int latentW = deriveLatentDim(latentFloats, 1); // W/8

        long[] latentShape = { 1, 4, latentH, latentW };
        long[] tsShape = { 1 };
        long[] embedShape = { 1, 77, 768 };

        // Timestep scalar tensor
        MemorySegment tsSeg = arena.allocate(Float.BYTES, 4);
        tsSeg.set(ValueLayout.JAVA_FLOAT, 0, timestep);

        MemorySegment tLatent = ort.createFloatTensor(latents, latentShape);
        MemorySegment tTs = ort.createFloatTensor(tsSeg, tsShape);
        MemorySegment tEmbed = ort.createFloatTensor(embeddings, embedShape);

        MemorySegment[] outputs = ort.run(
                session,
                new String[] { "sample", "timestep", "encoder_hidden_states" },
                new MemorySegment[] { tLatent, tTs, tEmbed },
                new String[] { "out_sample" });

        MemorySegment raw = ort.getTensorFloatData(outputs[0], latentFloats);
        MemorySegment owned = arena.allocate(latents.byteSize(), 8);
        owned.copyFrom(raw);

        ort.releaseValue(tLatent);
        ort.releaseValue(tTs);
        ort.releaseValue(tEmbed);
        ort.releaseValue(outputs[0]);

        return owned;
    }

    private int deriveLatentDim(long totalFloats, int dimIndex) {
        // totalFloats = 1 * 4 * H * W → H*W = totalFloats / 4
        // Assume square for simplicity; the real shape comes from the model
        long hw = totalFloats / 4;
        int side = (int) Math.round(Math.sqrt(hw));
        return side; // works for square latents (512→64, 768→96, etc.)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VaeDecoder
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * VAE latent decoder.
 *
 * Model: vae_decoder.onnx
 * Input: latent_sample [1, 4, H/8, W/8] float32
 * Output: sample [1, 3, H, W] float32 range ≈ [-1, 1]
 */
class VaeDecoder {

    private static final float VAE_SCALE = 0.18215f;

    private final OnnxRuntimeBridge ort;
    private final MemorySegment session;
    private final Arena arena;

    VaeDecoder(OnnxRuntimeBridge ort, Path modelPath, Arena arena) throws Throwable {
        this.ort = ort;
        this.arena = arena;
        this.session = ort.createSession(modelPath);
        System.out.println("[VaeDecoder] Loaded: " + modelPath);
    }

    /**
     * Decode latents to RGB image tensor.
     *
     * @param latents [1, 4, H/8, W/8] float tensor
     * @return [1, 3, H, W] float tensor, values ≈ [-1, 1]
     */
    MemorySegment decode(MemorySegment latents) throws Throwable {
        long latentFloats = latents.byteSize() / Float.BYTES;
        int hw = (int) Math.round(Math.sqrt(latentFloats / 4.0));

        // Scale latents before decoding (VAE convention)
        MemorySegment scaled = VectorizedOps.scale(latents, 1.0f / VAE_SCALE, arena);

        long[] shape = { 1, 4, hw, hw };
        MemorySegment tIn = ort.createFloatTensor(scaled, shape);

        MemorySegment[] outputs = ort.run(
                session,
                new String[] { "latent_sample" },
                new MemorySegment[] { tIn },
                new String[] { "sample" });

        long rgbFloats = 3L * hw * 8 * hw * 8; // 3 × H × W
        MemorySegment raw = ort.getTensorFloatData(outputs[0], rgbFloats);
        MemorySegment owned = arena.allocate(raw.byteSize(), 8);
        owned.copyFrom(raw);

        ort.releaseValue(tIn);
        ort.releaseValue(outputs[0]);

        return owned;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DDIMScheduler
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * DDIM (Denoising Diffusion Implicit Models) scheduler.
 *
 * Computes the linear noise schedule and implements the DDIM step rule:
 *
 * x_{t-1} = sqrt(α_{t-1}) * pred_x0
 * + sqrt(1 - α_{t-1} - σ²) * noise_pred
 * + σ * ε (σ=0 for deterministic DDIM)
 *
 * where pred_x0 = (x_t - sqrt(1-α_t)*noise_pred) / sqrt(α_t)
 */
class DDIMScheduler {

    private static final int TRAIN_STEPS = 1000;
    private static final float BETA_START = 0.00085f;
    private static final float BETA_END = 0.012f;

    private final int numSteps;
    private final float[] alphasCumprod; // ᾱ_t for t = 0..999
    private final List<Float> timestepList;

    DDIMScheduler(int numSteps) {
        this.numSteps = numSteps;
        this.alphasCumprod = computeAlphasCumprod();
        this.timestepList = buildTimesteps();
    }

    List<Float> timesteps() {
        return timestepList;
    }

    /**
     * Single DDIM denoising step (deterministic, σ=0).
     *
     * @param latents   x_t
     * @param noisePred ε_θ(x_t, t) from U-Net
     * @param t         current timestep
     * @param arena     for off-heap allocations
     * @return x_{t-1}
     */
    MemorySegment step(MemorySegment latents, MemorySegment noisePred,
            float t, Arena arena) {

        int tIdx = Math.round(t);
        int tPrev = Math.max(tIdx - (TRAIN_STEPS / numSteps), 0);

        float alphaCumprodT = alphasCumprod[tIdx];
        float alphaCumprodPrev = alphasCumprod[tPrev];

        float sqrtAlphaT = (float) Math.sqrt(alphaCumprodT);
        float sqrtOneMinusT = (float) Math.sqrt(1.0 - alphaCumprodT);
        float sqrtAlphaPrev = (float) Math.sqrt(alphaCumprodPrev);
        float sqrtDirPrev = (float) Math.sqrt(1.0 - alphaCumprodPrev); // σ=0

        // pred_x0 = (x_t - sqrt(1-α_t)*ε) / sqrt(α_t)
        int n = (int) (latents.byteSize() / Float.BYTES);
        MemorySegment predX0 = arena.allocate(latents.byteSize(), 8);
        for (int i = 0; i < n; i++) {
            float xt = latents.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float eps = noisePred.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            predX0.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    (xt - sqrtOneMinusT * eps) / sqrtAlphaT);
        }

        // x_{t-1} = sqrt(α_{t-1}) * pred_x0 + sqrt(1-α_{t-1}) * ε
        MemorySegment xPrev = VectorizedOps.add(
                VectorizedOps.scale(predX0, sqrtAlphaPrev, arena),
                VectorizedOps.scale(noisePred, sqrtDirPrev, arena),
                arena);

        return xPrev;
    }

    // ── Schedule computation ──────────────────────────────────────────────────

    private float[] computeAlphasCumprod() {
        // Scaled-linear (quadratic) beta schedule used by SD
        float[] betas = new float[TRAIN_STEPS];
        float sqrtStart = (float) Math.sqrt(BETA_START);
        float sqrtEnd = (float) Math.sqrt(BETA_END);
        for (int i = 0; i < TRAIN_STEPS; i++) {
            float beta = sqrtStart + (sqrtEnd - sqrtStart) * i / (TRAIN_STEPS - 1);
            betas[i] = beta * beta;
        }

        float[] alphas = new float[TRAIN_STEPS];
        float cumProd = 1.0f;
        float[] result = new float[TRAIN_STEPS];
        for (int i = 0; i < TRAIN_STEPS; i++) {
            alphas[i] = 1.0f - betas[i];
            cumProd *= alphas[i];
            result[i] = cumProd;
        }
        return result;
    }

    private List<Float> buildTimesteps() {
        // Evenly spaced timesteps in descending order, e.g. [981, 961, ..., 1]
        List<Float> ts = new ArrayList<>(numSteps);
        int step = TRAIN_STEPS / numSteps;
        for (int i = numSteps - 1; i >= 0; i--) {
            ts.add((float) (i * step));
        }
        return ts;
    }
}
