package tech.kayys.gollek.onnx.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.RunnerMetadata;
import tech.kayys.gollek.spi.model.ModalityType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.*;

/**
 * Stable Diffusion ONNX Runner for Gollek.
 *
 * <p>Orchestrates CLIP text encoder, UNet denoiser, and VAE decoder
 * via individual ONNX Runtime sessions using the FFM-based
 * {@link OnnxRuntimeBinding}.
 *
 * <h2>Pipeline</h2>
 * <pre>
 *   prompt → [CLIP text encoder] → text embeddings
 *   noise  → [UNet × N steps]    → denoised latents
 *   latents → [VAE decoder]      → RGB image
 * </pre>
 *
 * <h2>Memory Model</h2>
 * <p>All intermediate data is kept as raw {@link MemorySegment} buffers
 * managed by a shared {@link Arena}. Buffers are wrapped into OrtValues
 * <em>only</em> at the point of each {@code ort.run()} call and released
 * immediately after, preventing use-after-free crashes in native code.
 */
@ApplicationScoped
public class StableDiffusionOnnxRunner extends AbstractGollekRunner {

    private static final Logger LOG = Logger.getLogger(StableDiffusionOnnxRunner.class);
    public static final String RUNNER_NAME = "sd-onnx";

    // ── Latent space constants ───────────────────────────────────────────────
    private static final int LATENT_CHANNELS = 4;
    private static final int LATENT_SIZE = 64;   // 512 / 8
    private static final int IMAGE_SIZE = 512;
    private static final int IMAGE_CHANNELS = 3;
    private static final long NUM_LATENT_FLOATS = 1L * LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE;
    private static final long LATENT_BYTES = NUM_LATENT_FLOATS * Float.BYTES;
    private static final int CLIP_SEQ_LEN = 77;
    private static final int CLIP_HIDDEN_DIM = 768;
    private static final long TEXT_EMBED_FLOATS = 1L * CLIP_SEQ_LEN * CLIP_HIDDEN_DIM;
    private static final long TEXT_EMBED_BYTES = TEXT_EMBED_FLOATS * Float.BYTES;

    // ── DDIM scheduler constants ─────────────────────────────────────────────
    private static final int TOTAL_TRAIN_TIMESTEPS = 1000;
    private static final float BETA_START = 0.00085f;
    private static final float BETA_END = 0.012f;
    /** Pre-computed alpha_bar table (avoids O(t) loop per step). */
    private final double[] alphaBarTable;

    @ConfigProperty(name = "gollek.runners.onnx.library-path", defaultValue = "/usr/lib/libonnxruntime.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.onnx.execution-provider", defaultValue = "auto")
    String executionProvider;

    private OnnxRuntimeBinding ort;
    private MemorySegment ortEnv = MemorySegment.NULL;
    private MemorySegment memInfo = MemorySegment.NULL;

    private Tokenizer tokenizer;

    // Sessions for pipeline components
    private MemorySegment textEncoderSession = MemorySegment.NULL;
    private MemorySegment unetSession = MemorySegment.NULL;
    private MemorySegment vaeDecoderSession = MemorySegment.NULL;

    // Discovered tensor names for each session
    private String[] textEncoderInputNames, textEncoderOutputNames;
    private String[] unetInputNames, unetOutputNames;
    private String[] vaeInputNames, vaeOutputNames;

    private String resolvedEp = "cpu";
    private ModelManifest manifest;

    public StableDiffusionOnnxRunner() {
        // Pre-compute alpha_bar table once
        alphaBarTable = new double[TOTAL_TRAIN_TIMESTEPS + 1];
        alphaBarTable[0] = 1.0 - BETA_START;
        for (int i = 1; i <= TOTAL_TRAIN_TIMESTEPS; i++) {
            double beta = BETA_START + (BETA_END - BETA_START) * (double) i / (TOTAL_TRAIN_TIMESTEPS - 1);
            alphaBarTable[i] = alphaBarTable[i - 1] * (1.0 - beta);
        }
    }

    // ── ModelRunner identity ─────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "onnxruntime";
    }

    @Override
    public DeviceType deviceType() {
        return executionProvider.toLowerCase().contains("coreml") || executionProvider.equals("auto")
                ? DeviceType.METAL : DeviceType.CPU;
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.ONNX),
                List.of(DeviceType.CPU, DeviceType.METAL),
                Map.of("pipeline", "stable-diffusion-v1"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportedDataTypes(new String[]{"fp32", "fp16"})
                .build();
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        this.manifest = modelManifest;

        // 1. Load native library
        Path libPath = resolveLibraryPath(libraryPath);
        OnnxRuntimeBinding.initialize(libPath);
        ort = OnnxRuntimeBinding.getInstance();

        if (!ort.isNativeAvailable()) {
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "Native ONNX Runtime not available at " + libPath);
        }

        // 2. Create ORT environment and CPU memory info
        ortEnv = ort.createEnv("gollek-sd");
        memInfo = ort.createCpuMemoryInfo();

        // 3. Configure separate options for Hybrid Backend
        MemorySegment metalOpts = ort.createSessionOptions();
        ort.setIntraOpNumThreads(metalOpts, 2);
        String ep = resolveAndAttachEp(metalOpts);

        // CPU-only options for the heavily-fragmented UNet to prevent system hangs
        MemorySegment unetOpts = ort.createSessionOptions();
        ort.setIntraOpNumThreads(unetOpts, 2);

        // 4. Load all three pipeline sessions
        Path baseDir = resolveBaseDir(modelManifest);
        LOG.infof("[SD-ONNX] Base model directory: %s", baseDir);

        try {
            Path tePath = resolveSubmodel(baseDir, "text_encoder");
            Path unetPath = resolveSubmodel(baseDir, "unet");
            Path vaePath = resolveSubmodel(baseDir, "vae_decoder");

            LOG.infof("[SD-ONNX] Loading text encoder (Metal): %s", tePath);
            textEncoderSession = ort.createSession(ortEnv, tePath.toString(), metalOpts);
            textEncoderInputNames = discoverInputNames(textEncoderSession);
            textEncoderOutputNames = discoverOutputNames(textEncoderSession);

            LOG.infof("[SD-ONNX] Loading UNet (CPU-Safe): %s", unetPath);
            unetSession = ort.createSession(ortEnv, unetPath.toString(), unetOpts);
            unetInputNames = discoverInputNames(unetSession);
            unetOutputNames = discoverOutputNames(unetSession);

            // Tokenizer
            Path tokenizerPath = baseDir.resolve("tokenizer");
            try {
                this.tokenizer = TokenizerFactory.load(tokenizerPath, null);
                LOG.infof("[SD-ONNX] Tokenizer loaded from %s", tokenizerPath);
            } catch (Exception e) {
                LOG.errorf("[SD-ONNX] Failed to load CLIP tokenizer: %s", e.getMessage());
                throw new RuntimeException("CLIP tokenizer required", e);
            }

            LOG.infof("[SD-ONNX] Loading VAE decoder (Metal): %s", vaePath);
            vaeDecoderSession = ort.createSession(ortEnv, vaePath.toString(), metalOpts);
            vaeInputNames = discoverInputNames(vaeDecoderSession);
            vaeOutputNames = discoverOutputNames(vaeDecoderSession);

        } catch (Exception e) {
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "Failed to load SD components: " + e.getMessage());
        } finally {
            ort.releaseSessionOptions(metalOpts);
            ort.releaseSessionOptions(unetOpts);
        }

        this.initialized = true;
        this.resolvedEp = ep;
        LOG.infof("[SD-ONNX] Pipeline ready — EP=%s, UNet inputs=%s, outputs=%s",
                resolvedEp,
                Arrays.toString(unetInputNames),
                Arrays.toString(unetOutputNames));
    }

    // ── Inference ────────────────────────────────────────────────────────────

    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        throw new UnsupportedOperationException(
                "Use stream() for Stable Diffusion to get progress updates and image chunks.");
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                String reqId = request.getRequestId();
                String prompt = Objects.requireNonNullElse(request.getPrompt(), "");

                int steps = paramInt(request, "steps", 20);
                float guidance = paramFloat(request, "guidance_scale", 7.5f);
                long seed = paramLong(request, "seed", 42L);
                int width = paramInt(request, "width", IMAGE_SIZE);
                int height = paramInt(request, "height", IMAGE_SIZE);

                // Validate dimensions
                if (width % 64 != 0 || height % 64 != 0) {
                    LOG.warnf("[SD-ONNX] Invalid dimensions %dx%d, must be multiples of 64. Using default 512x512.", 
                            width, height);
                    width = IMAGE_SIZE;
                    height = IMAGE_SIZE;
                }
                
                if (width != IMAGE_SIZE || height != IMAGE_SIZE) {
                    LOG.warnf("[SD-ONNX] Non-default resolution %dx%d requested. This runner uses fixed-shape ONNX models optimized for 512x512. Results may be degraded.", 
                            width, height);
                }

                LOG.infof("[SD-ONNX] Generating — prompt='%s', steps=%d, guidance=%.1f, seed=%d, size=%dx%d",
                        prompt, steps, guidance, seed, width, height);

                try (Arena arena = Arena.ofShared()) {

                    // 1. Text Encoding — run CLIP on prompt and empty string
                    emitter.emit(progressChunk(reqId, 0, 
                            String.format("[0/%d] Encoding prompt...", steps + 2)));
                    MemorySegment textEmbedData = encodePrompt(arena, prompt);
                    MemorySegment nullEmbedData = encodePrompt(arena, "");

                    // 2. Latent Initialization — Gaussian noise [1, 4, 64, 64]
                    emitter.emit(progressChunk(reqId, 1, 
                            String.format("[1/%d] Initializing latents (seed=%d)...", steps + 2, seed)));
                    MemorySegment latentData = createNoiseLatents(arena, seed);

                    // 3. Diffusion Loop
                    for (int i = 0; i < steps; i++) {
                        emitter.emit(progressChunk(reqId, i + 2,
                                String.format("[%d/%d] Denoising step %d/%d (%.0f%%)...", 
                                        i + 2, steps + 2, i + 1, steps, 
                                        ((i + 1.0) / steps) * 100.0)));
                        latentData = denoiseStep(arena, latentData, textEmbedData,
                                nullEmbedData, i, steps, guidance);
                    }

                    // 4. VAE Decoding — latents → RGB pixels → PNG
                    emitter.emit(progressChunk(reqId, steps + 2, 
                            String.format("[%d/%d] Decoding image to PNG...", steps + 2, steps + 2)));
                    byte[] pngData = decodeToPng(arena, latentData);

                    // 5. Emit result
                    String base64 = Base64.getEncoder().encodeToString(pngData);
                    emitter.emit(StreamingInferenceChunk.imageChunk(
                            reqId, steps + 3, base64, true));
                    emitter.complete();
                }

            } catch (Exception e) {
                LOG.error("[SD-ONNX] Generation failed", e);
                emitter.fail(e);
            }
        });
    }

    // ── Internal Pipeline Logic ──────────────────────────────────────────────

    /**
     * Run CLIP text encoder. Returns raw FP32 buffer of shape [1, 77, 768].
     */
    private MemorySegment encodePrompt(Arena arena, String prompt) {
        // 1. Tokenize
        EncodeOptions opts = new EncodeOptions();
        opts.addBos = true;
        opts.addEos = true;
        long[] tokens = tokenizer.encode(prompt, opts);

        // CLIP fixed sequence length for SD v1 is 77
        long[] finalTokens = new long[CLIP_SEQ_LEN];
        for (int i = 0; i < CLIP_SEQ_LEN; i++) {
            if (i < tokens.length) {
                finalTokens[i] = tokens[i];
            } else {
                finalTokens[i] = tokenizer.padTokenId();
            }
        }

        // 2. Wrap as OrtValue [1, 77] (INT32 required by CLIP ONNX models)
        MemorySegment inputData = arena.allocate(CLIP_SEQ_LEN * 4L, 4);
        for (int i = 0; i < CLIP_SEQ_LEN; i++) {
            inputData.setAtIndex(ValueLayout.JAVA_INT, i, (int) finalTokens[i]);
        }

        MemorySegment inputVal = ort.createTensorWithData(memInfo, inputData,
                new long[]{1, CLIP_SEQ_LEN},
                OnnxRuntimeBinding.ONNX_TENSOR_INT32);

        try {
            // Text encoder session input order mapping
            String[] names = {"input_ids"};
            MemorySegment[] vals = {inputVal};
            MemorySegment[] orderedVals = new MemorySegment[textEncoderInputNames.length];
            for (int i = 0; i < textEncoderInputNames.length; i++) {
                String actualName = textEncoderInputNames[i];
                boolean found = false;
                for (int j = 0; j < names.length; j++) {
                    if (actualName.equalsIgnoreCase(names[j]) || actualName.contains("input")) {
                        orderedVals[i] = vals[j];
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    LOG.warnf("[SD-ONNX] Text Encoder unknown input: %s. Defaulting to input_ids.", actualName);
                    orderedVals[i] = inputVal;
                }
            }

            MemorySegment[] results = ort.run(
                    textEncoderSession, MemorySegment.NULL,
                    textEncoderInputNames,
                    orderedVals,
                    textEncoderOutputNames);

            try {
                // Extract last_hidden_state [1, 77, 768]
                MemorySegment srcPtr = ort.getTensorMutableData(results[0]);
                MemorySegment srcData = srcPtr.reinterpret(TEXT_EMBED_BYTES);
                MemorySegment outputData = arena.allocate(TEXT_EMBED_BYTES, 4);
                outputData.copyFrom(srcData);
                return outputData;
            } finally {
                for (MemorySegment r : results) ort.releaseValue(r);
            }
        } finally {
            ort.releaseValue(inputVal);
        }
    }

    /**
     * Creates random Gaussian noise in latent space [1, 4, 64, 64].
     * Returns a raw FP32 buffer.
     * 
     * @param arena the memory arena for allocation
     * @param seed the random seed for reproducible generation
     * @return noise latent tensor
     */
    private MemorySegment createNoiseLatents(Arena arena, long seed) {
        MemorySegment data = arena.allocate(LATENT_BYTES, 4);
        Random rnd = new Random(seed);
        for (long i = 0; i < NUM_LATENT_FLOATS; i++) {
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) rnd.nextGaussian());
        }
        return data;
    }

    /**
     * One denoising step: runs UNet twice (cond + uncond) and applies
     * classifier-free guidance + DDIM scheduler.
     *
     * <p>All data stays as raw MemorySegment buffers. OrtValues are created
     * transiently for each {@code ort.run()} call and released immediately.
     *
     * @param latentData  raw FP32 [1,4,64,64]
     * @param textEmbeds  raw FP32 [1,77,768] for the prompt
     * @param nullEmbeds  raw FP32 [1,77,768] for empty prompt
     * @return updated latentData (raw FP32 [1,4,64,64])
     */
    private MemorySegment denoiseStep(Arena arena, MemorySegment latentData,
                                       MemorySegment textEmbeds, MemorySegment nullEmbeds,
                                       int step, int totalSteps, float guidance) {

        // Timestep for this DDIM step (INT64 scalar)
        long timestep = computeTimestep(step, totalSteps);
        MemorySegment timestepData = arena.allocate(8L, 8);
        timestepData.set(ValueLayout.JAVA_LONG, 0, timestep);

        // ─── Sequential UNet passes (Safe Mode) ──────────────────────────
        // We use sequential passes to reduce peak CPU load and memory pressure.
        MemorySegment noiseCond = runUNet(arena, latentData, timestepData, textEmbeds, 1);
        MemorySegment noiseUncond = runUNet(arena, latentData, timestepData, nullEmbeds, 1);

        // ─── Classifier-free guidance ────────────────────────────────────
        // noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
        MemorySegment noisePred = arena.allocate(LATENT_BYTES, 4);
        for (long i = 0; i < NUM_LATENT_FLOATS; i++) {
            float c = noiseCond.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float u = noiseUncond.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            noisePred.setAtIndex(ValueLayout.JAVA_FLOAT, i, u + guidance * (c - u));
        }

        // ─── DDIM scheduler step ─────────────────────────────────────────
        return applyDDIMStep(arena, latentData, noisePred, step, totalSteps);
    }


    /**
     * Runs the UNet session with properly wrapped OrtValue inputs.
     * Returns raw FP32 buffer with the noise prediction [1,4,64,64].
     */
    private MemorySegment runUNet(Arena arena, MemorySegment latentData,
                                   MemorySegment timestepData, MemorySegment embedData,
                                   int batchSize) {

        // Wrap raw data as OrtValues
        MemorySegment latentVal = ort.createTensorWithData(memInfo, latentData,
                new long[]{batchSize, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE},
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT);

        MemorySegment timestepVal = ort.createTensorWithData(memInfo, timestepData,
                new long[]{1},  // scalar as [1]
                OnnxRuntimeBinding.ONNX_TENSOR_INT64);

        MemorySegment embedVal = ort.createTensorWithData(memInfo, embedData,
                new long[]{batchSize, CLIP_SEQ_LEN, CLIP_HIDDEN_DIM},
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT);

        try {
            // Robustly map inputs by name to ensure correct order
            String[] names = {"sample", "timestep", "encoder_hidden_states"};
            MemorySegment[] vals = {latentVal, timestepVal, embedVal};
            
            MemorySegment[] orderedVals = new MemorySegment[unetInputNames.length];
            for (int i = 0; i < unetInputNames.length; i++) {
                String actualName = unetInputNames[i];
                boolean found = false;
                for (int j = 0; j < names.length; j++) {
                    if (actualName.equalsIgnoreCase(names[j])) {
                        orderedVals[i] = vals[j];
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw new RuntimeException("UNet session requires unknown input: " + actualName);
                }
            }

            MemorySegment[] results = ort.run(
                    unetSession, MemorySegment.NULL,
                    unetInputNames,
                    orderedVals,
                    unetOutputNames);

            // Extract raw data from output OrtValue
            long outputBytes = LATENT_BYTES * batchSize;
            MemorySegment outputData = arena.allocate(outputBytes, 4);
            MemorySegment srcPtr = ort.getTensorMutableData(results[0]);
            MemorySegment srcData = srcPtr.reinterpret(outputBytes);
            outputData.copyFrom(srcData);

            // Release output OrtValues
            for (MemorySegment r : results) {
                ort.releaseValue(r);
            }

            return outputData;
        } finally {
            // Always release input OrtValues (data buffers are arena-managed)
            ort.releaseValue(latentVal);
            ort.releaseValue(timestepVal);
            ort.releaseValue(embedVal);
        }
    }

    /**
     * VAE decode: latent [1,4,64,64] → image [1,3,512,512] → PNG bytes.
     */
    private byte[] decodeToPng(Arena arena, MemorySegment latentData) {
        try {
            // Scale latents by 1/0.18215 (SD v1 VAE scaling factor)
            MemorySegment scaledLatents = arena.allocate(LATENT_BYTES, 4);
            float scale = 1.0f / 0.18215f;
            for (long i = 0; i < NUM_LATENT_FLOATS; i++) {
                scaledLatents.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                        latentData.getAtIndex(ValueLayout.JAVA_FLOAT, i) * scale);
            }

            // Wrap as OrtValue
            MemorySegment latentVal = ort.createTensorWithData(memInfo, scaledLatents,
                    new long[]{1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE},
                    OnnxRuntimeBinding.ONNX_TENSOR_FLOAT);

            try {
                // Robustly map VAE inputs
                String[] names = {"latent_sample", "latent"};
                MemorySegment[] vals = {latentVal, latentVal}; // try both names
                MemorySegment[] orderedVals = new MemorySegment[vaeInputNames.length];
                for (int i = 0; i < vaeInputNames.length; i++) {
                    String actualName = vaeInputNames[i];
                    boolean found = false;
                    for (int j = 0; j < names.length; j++) {
                        if (actualName.equalsIgnoreCase(names[j]) || actualName.contains("latent")) {
                            orderedVals[i] = vals[j];
                            found = true;
                            break;
                        }
                    }
                    if (!found) orderedVals[i] = latentVal; // default
                }

                MemorySegment[] results = ort.run(
                        vaeDecoderSession, MemorySegment.NULL,
                        vaeInputNames,
                        orderedVals,
                        vaeOutputNames);

                // Extract decoded image data
                long imageFloats = 1L * IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;
                long imageBytes = imageFloats * Float.BYTES;
                MemorySegment imageData = arena.allocate(imageBytes, 4);
                MemorySegment srcPtr = ort.getTensorMutableData(results[0]);
                MemorySegment srcData = srcPtr.reinterpret(imageBytes);
                imageData.copyFrom(srcData);

                for (MemorySegment r : results) {
                    ort.releaseValue(r);
                }

                return encodeNchwToPng(imageData, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS);

            } finally {
                ort.releaseValue(latentVal);
            }

        } catch (Exception e) {
            LOG.warnf("[SD-ONNX] VAE decode failed: %s. Returning gradient placeholder.", e.getMessage());
            return generateGradientPng();
        }
    }

    // ── DDIM Scheduler ───────────────────────────────────────────────────────

    private long computeTimestep(int step, int totalSteps) {
        double stepRatio = (double) TOTAL_TRAIN_TIMESTEPS / totalSteps;
        return (long) ((totalSteps - 1 - step) * stepRatio);
    }

    /**
     * DDIM deterministic step (η=0).
     * <pre>
     * x_prev = √α̅_{t-1} · pred_x0  +  √(1-α̅_{t-1}) · pred_dir
     * where pred_x0 = (x_t - √(1-α̅_t) · ε) / √α̅_t
     * </pre>
     */
    private MemorySegment applyDDIMStep(Arena arena, MemorySegment latentData,
                                         MemorySegment noisePred, int step, int totalSteps) {

        double stepRatio = (double) TOTAL_TRAIN_TIMESTEPS / totalSteps;
        int tCurrent = (int) ((totalSteps - 1 - step) * stepRatio);
        int tPrev = Math.max(0, (int) ((totalSteps - 2 - step) * stepRatio));

        double aBarT = alphaBarTable[tCurrent];
        double aBarPrev = (step + 1 >= totalSteps) ? 1.0 : alphaBarTable[tPrev];

        double sqrtABarT = Math.sqrt(aBarT);
        double sqrtOneMinusABarT = Math.sqrt(1.0 - aBarT);
        double sqrtABarPrev = Math.sqrt(aBarPrev);
        double sqrtOneMinusABarPrev = Math.sqrt(1.0 - aBarPrev);

        MemorySegment newLatent = arena.allocate(LATENT_BYTES, 4);
        for (long i = 0; i < NUM_LATENT_FLOATS; i++) {
            double x = latentData.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            double eps = noisePred.getAtIndex(ValueLayout.JAVA_FLOAT, i);

            // Predict x_0
            double predX0 = (x - sqrtOneMinusABarT * eps) / sqrtABarT;
            // Direction pointing to x_t
            double predDir = sqrtOneMinusABarPrev * eps;
            // DDIM update
            double xPrev = sqrtABarPrev * predX0 + predDir;

            newLatent.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) xPrev);
        }
        return newLatent;
    }

    // ── PNG encoding ─────────────────────────────────────────────────────────

    /**
     * Encodes NCHW FP32 image data to PNG.
     * Assumes values are in [-1, 1] (SD VAE output) and maps to [0, 255].
     */
    private byte[] encodeNchwToPng(MemorySegment imageData, int width, int height, int channels) {
        try (java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream()) {
            // PNG signature
            baos.write(new byte[]{(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A});

            // IHDR
            writePngChunk(baos, "IHDR", new byte[]{
                    (byte) (width >> 24), (byte) (width >> 16), (byte) (width >> 8), (byte) width,
                    (byte) (height >> 24), (byte) (height >> 16), (byte) (height >> 8), (byte) height,
                    8, 2, 0, 0, 0 // 8-bit RGB, no compression/filter/interlace
            });

            // IDAT — convert NCHW to row-major RGB
            int bytesPerPixel = 3; // RGB
            int rowSize = width * bytesPerPixel + 1; // +1 for filter byte
            byte[] rawData = new byte[rowSize * height];

            long channelStride = (long) width * height; // NCHW: each channel is W*H contiguous
            for (int y = 0; y < height; y++) {
                rawData[y * rowSize] = 0; // filter: none
                for (int x = 0; x < width; x++) {
                    int dstIdx = y * rowSize + 1 + x * bytesPerPixel;
                    for (int c = 0; c < 3; c++) {
                        // NCHW layout: index = c * H * W + y * W + x
                        long srcIdx = c * channelStride + (long) y * width + x;
                        float val = imageData.getAtIndex(ValueLayout.JAVA_FLOAT, srcIdx);
                        // SD VAE outputs [-1, 1]; map to [0, 255]
                        int byteVal = (int) ((val + 1.0f) * 0.5f * 255.0f);
                        rawData[dstIdx + c] = (byte) Math.max(0, Math.min(255, byteVal));
                    }
                }
            }

            // Compress with zlib
            java.util.zip.Deflater deflater = new java.util.zip.Deflater();
            deflater.setInput(rawData);
            deflater.finish();
            byte[] compressed = new byte[rawData.length + 1024];
            int compressedSize = deflater.deflate(compressed);
            deflater.end();

            byte[] idatPayload = new byte[compressedSize];
            System.arraycopy(compressed, 0, idatPayload, 0, compressedSize);
            writePngChunk(baos, "IDAT", idatPayload);

            // IEND
            writePngChunk(baos, "IEND", new byte[0]);

            return baos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException("PNG encoding failed", e);
        }
    }

    private void writePngChunk(java.io.OutputStream out, String type, byte[] data) throws Exception {
        java.util.zip.CRC32 crc = new java.util.zip.CRC32();
        byte[] typeBytes = type.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        crc.update(typeBytes);
        crc.update(data);

        out.write((data.length >> 24) & 0xFF);
        out.write((data.length >> 16) & 0xFF);
        out.write((data.length >> 8) & 0xFF);
        out.write(data.length & 0xFF);
        out.write(typeBytes);
        out.write(data);
        long crcVal = crc.getValue();
        out.write((int) (crcVal >> 24) & 0xFF);
        out.write((int) (crcVal >> 16) & 0xFF);
        out.write((int) (crcVal >> 8) & 0xFF);
        out.write((int) crcVal & 0xFF);
    }

    /**
     * Gradient placeholder PNG for when VAE decode fails.
     */
    private byte[] generateGradientPng() {
        int w = 64, h = 64;
        long numFloats = (long) IMAGE_CHANNELS * w * h;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment img = a.allocate(numFloats * Float.BYTES, 4);
            // NCHW gradient: R=x/w, G=y/h, B=0.5
            long stride = (long) w * h;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    long idx = (long) y * w + x;
                    img.setAtIndex(ValueLayout.JAVA_FLOAT, idx, (float) x / w * 2 - 1);                   // R
                    img.setAtIndex(ValueLayout.JAVA_FLOAT, stride + idx, (float) y / h * 2 - 1);           // G
                    img.setAtIndex(ValueLayout.JAVA_FLOAT, 2 * stride + idx, 0.0f);                        // B
                }
            }
            return encodeNchwToPng(img, w, h, IMAGE_CHANNELS);
        } catch (Exception e) {
            // Absolute last-resort: 1×1 transparent PNG
            return Base64.getDecoder().decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==");
        }
    }

    // ── Session helpers ──────────────────────────────────────────────────────

    private String[] discoverInputNames(MemorySegment session) {
        int count = (int) ort.getInputCount(session);
        String[] names = new String[count];
        for (int i = 0; i < count; i++) names[i] = ort.getInputName(session, i);
        return names;
    }

    private String[] discoverOutputNames(MemorySegment session) {
        int count = (int) ort.getOutputCount(session);
        String[] names = new String[count];
        for (int i = 0; i < count; i++) names[i] = ort.getOutputName(session, i);
        return names;
    }

    // ── Path resolution ──────────────────────────────────────────────────────

    private Path resolveBaseDir(ModelManifest manifest) {
        return manifest.artifacts().values().stream()
                .findFirst()
                .map(loc -> {
                    String uri = loc.uri();
                    if (uri.startsWith("file:")) return Path.of(java.net.URI.create(uri));
                    return Path.of(uri);
                })
                .map(p -> Files.isDirectory(p) ? p : p.getParent())
                .orElseThrow(() -> new RuntimeException("No artifact URI in manifest"));
    }

    /**
     * Resolves an ONNX submodel path. Tries the conventional subdirectory
     * name first, then scans for model.onnx in alternate locations.
     */
    private Path resolveSubmodel(Path baseDir, String subdir) {
        Path expected = baseDir.resolve(subdir).resolve("model.onnx");
        if (Files.exists(expected)) return expected;

        // Fallback: look one level up for the blob directory
        Path parent = baseDir.getParent();
        if (parent != null) {
            Path alt = parent.resolve(subdir).resolve("model.onnx");
            if (Files.exists(alt)) return alt;
        }

        throw new RuntimeException("Cannot find " + subdir + "/model.onnx under " + baseDir);
    }

    private String resolveAndAttachEp(MemorySegment opts) {
        if (executionProvider.equals("auto") || executionProvider.contains("coreml")) {
            if (ort.appendCoreMlProvider(opts, 0)) return "CoreMLExecutionProvider";
        }
        return "CPUExecutionProvider";
    }

    private Path resolveLibraryPath(String configuredPath) {
        Path configured = Path.of(configuredPath);
        if (Files.exists(configured)) return configured;
        // Common fallback paths
        Path[] fallbacks = {
            Path.of(System.getProperty("user.home"), ".gollek", "libs", "libonnxruntime.dylib"),
            Path.of(System.getProperty("user.home"), ".gollek", "libs", "onnxruntime", "libonnxruntime.dylib"),
            Path.of("/opt/homebrew/lib/libonnxruntime.dylib"),
            Path.of("/usr/local/lib/libonnxruntime.dylib"),
        };
        for (Path p : fallbacks) {
            if (Files.exists(p)) return p;
        }
        return fallbacks[0]; // Default fallback
    }

    // ── Parameter extraction helpers ─────────────────────────────────────────

    private static int paramInt(InferenceRequest req, String key, int defaultVal) {
        Object v = req.getParameters().get(key);
        return v instanceof Number n ? n.intValue() : defaultVal;
    }

    private static float paramFloat(InferenceRequest req, String key, float defaultVal) {
        Object v = req.getParameters().get(key);
        return v instanceof Number n ? n.floatValue() : defaultVal;
    }

    private static long paramLong(InferenceRequest req, String key, long defaultVal) {
        Object v = req.getParameters().get(key);
        return v instanceof Number n ? n.longValue() : defaultVal;
    }

    private static StreamingInferenceChunk progressChunk(String reqId, int index, String text) {
        return StreamingInferenceChunk.textDelta(reqId, index, text);
    }

    // ── Health & close ───────────────────────────────────────────────────────

    @Override
    public boolean health() {
        return this.initialized;
    }

    @Override
    public void close() {
        if (ort != null) {
            if (!textEncoderSession.equals(MemorySegment.NULL)) ort.releaseSession(textEncoderSession);
            if (!unetSession.equals(MemorySegment.NULL)) ort.releaseSession(unetSession);
            if (!vaeDecoderSession.equals(MemorySegment.NULL)) ort.releaseSession(vaeDecoderSession);
            if (!ortEnv.equals(MemorySegment.NULL)) ort.releaseEnv(ortEnv);
            if (!memInfo.equals(MemorySegment.NULL)) ort.releaseMemoryInfo(memInfo);
        }
        this.initialized = false;
    }
}
