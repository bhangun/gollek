package tech.kayys.gollek.cuda.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.plugin.runner.RunnerInitializationException;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.cuda.binding.CudaBinding;
import tech.kayys.gollek.cuda.config.CudaRunnerMode;
import tech.kayys.gollek.cuda.detection.CudaDetector;
import tech.kayys.gollek.cuda.detection.CudaCapabilities;
import tech.kayys.gollek.cuda.optimization.CudaOptimizationManager;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.RunnerMetadata;

import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * NVIDIA CUDA ModelRunner for Gollek.
 *
 * <p>Runs {@link ModelFormat#GGUF} and {@link ModelFormat#SAFETENSORS} models
 * in-process on NVIDIA GPUs via {@link CudaBinding}, which wraps the
 * CUDA Driver API ({@code cuda.h}) through Java FFM.
 *
 * <h2>Kernel pipeline</h2>
 * <p>This runner loads a pre-compiled CUDA kernel module ({@code .cubin} file)
 * compiled from custom CUDA kernels. The forward pass follows the same structure as
 * {@link tech.kayys.gollek.extension.metal.runner.MetalRunner}:
 * <ol>
 *   <li>RMS Norm — {@code gollek_cuda_rmsnorm}</li>
 *   <li>QKV projection — {@code gollek_cuda_matmul} (cuBLAS/cuBLASLt)</li>
 *   <li>FlashAttention-2/3 — {@code gollek_cuda_flash_attn_v2/v3}</li>
 *   <li>SiLU FFN — {@code gollek_cuda_silu_ffn}</li>
 * </ol>
 *
 * <h2>FlashAttention Support</h2>
 * <p>On A100+ (compute capability ≥ 8.0), FlashAttention-2 is used for
 * fused QK^T/softmax/×V in a single GPU pass. On H100+ (≥ 9.0),
 * FlashAttention-3 adds FP8 tensor core acceleration.
 *
 * <h2>Memory Model</h2>
 * <p>On A100/H100 with unified memory architecture, {@link CudaBinding#mallocManaged}
 * allocates memory accessible from both CPU and GPU without explicit copies.
 * On older GPUs (V100, RTX series), explicit H2D/D2H copies are performed.
 *
 * <h2>Compatibility</h2>
 * <p>Tested target architectures:
 * <ul>
 *   <li>sm_80 — NVIDIA A100 (40/80 GB HBM2e)</li>
 *   <li>sm_90 — NVIDIA H100 (80 GB HBM3)</li>
 *   <li>sm_86 — NVIDIA RTX A6000 (48 GB GDDR6)</li>
 *   <li>sm_89 — NVIDIA RTX 4090 (24 GB GDDR6X)</li>
 * </ul>
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.runners.cuda.enabled=false
 *   gollek.runners.cuda.mode=auto  # auto|standard|offload|force|disabled
 *   gollek.runners.cuda.library-path=/usr/local/cuda/lib64/libgollek_cuda.so
 *   gollek.runners.cuda.device-id=0
 *   gollek.runners.cuda.num-layers=32
 *   gollek.runners.cuda.num-heads=32
 *   gollek.runners.cuda.num-heads-kv=8
 *   gollek.runners.cuda.head-dim=128
 *   gollek.runners.cuda.model-dim=4096
 *   gollek.runners.cuda.ffn-dim=14336
 *   gollek.runners.cuda.vocab-size=32000
 * </pre>
 *
 * <h3>Build CUDA kernels</h3>
 * <pre>
 *   make -C src/main/cpp/cuda CUDA_ARCH=sm_80
 *   # Output: target/native/linux-x86_64/gollek_cuda_sm80.cubin
 * </pre>
 */
@ApplicationScoped
public class CudaRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "cuda";

    @ConfigProperty(name = "gollek.runners.cuda.enabled",      defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.cuda.mode",         defaultValue = "auto")
    String cudaMode;

    @ConfigProperty(name = "gollek.runners.cuda.library-path",
                    defaultValue = "/usr/local/cuda/lib64/libgollek_cuda.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.cuda.device-id",    defaultValue = "0")
    int deviceId;

    @ConfigProperty(name = "gollek.runners.cuda.num-layers",   defaultValue = "32")
    int numLayers;

    @ConfigProperty(name = "gollek.runners.cuda.num-heads",    defaultValue = "32")
    int numHeads;

    @ConfigProperty(name = "gollek.runners.cuda.num-heads-kv", defaultValue = "8")
    int numHeadsKv;

    @ConfigProperty(name = "gollek.runners.cuda.head-dim",     defaultValue = "128")
    int headDim;

    @ConfigProperty(name = "gollek.runners.cuda.model-dim",    defaultValue = "4096")
    int modelDim;

    @ConfigProperty(name = "gollek.runners.cuda.ffn-dim",      defaultValue = "14336")
    int ffnDim;

    @ConfigProperty(name = "gollek.runners.cuda.vocab-size",   defaultValue = "32000")
    int vocabSize;

    @Inject
    PagedKVCacheManager kvCacheManager;

    @Inject
    CudaDetector detector;

    private CudaBinding cuda;
    private CudaOptimizationManager optimizationManager;
    private CudaCapabilities caps;
    private ModelManifest manifest;
    private MemorySegment weightsMem;
    private Arena weightsArena;
    private MemorySegment[] layerSlices;

    // Device memory
    private MemorySegment dResidual = MemorySegment.NULL;
    private MemorySegment dNormed = MemorySegment.NULL;
    private MemorySegment dQkv = MemorySegment.NULL;
    private MemorySegment dAttnOut = MemorySegment.NULL;
    private MemorySegment dFfnBuf = MemorySegment.NULL;
    private MemorySegment cudaStream = MemorySegment.NULL;
    private boolean isUnifiedMemory = false;

    private static Path resolveLibraryPath(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return Path.of("");
        }
        if (rawPath.startsWith("~/")) {
            String home = System.getProperty("user.home");
            if (home != null && !home.isBlank()) {
                return Path.of(home + rawPath.substring(1));
            }
        }
        return Path.of(rawPath);
    }

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "cuda";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public RunnerMetadata metadata() {
        int computeCap = caps != null ? caps.cudaComputeCap() : 0;
        boolean hasFA2 = computeCap >= 80;
        boolean hasFA3 = computeCap >= 90;
        String attnPath = hasFA3 ? "FlashAttention-3-FP8"
                : hasFA2 ? "FlashAttention-2" : "cuBLAS-matmul";
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.GGUF, ModelFormat.SAFETENSORS),
                List.of(DeviceType.CUDA, DeviceType.CPU),
                Map.ofEntries(
                        Map.entry("cuda_api", "12.x"),
                        Map.entry("unified_memory", String.valueOf(isUnifiedMemory)),
                        Map.entry("device", caps != null ? caps.deviceName() : "N/A"),
                        Map.entry("compute_cap", String.valueOf(computeCap)),
                        Map.entry("attention_path", attnPath),
                        Map.entry("fa2_support", String.valueOf(hasFA2)),
                        Map.entry("fa3_support", String.valueOf(hasFA3)),
                        Map.entry("fp8_support", String.valueOf(hasFA3))
                )
        );
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true)
                .maxBatchSize(32)
                .supportedDataTypes(new String[] { "fp32", "fp16", "bf16", "fp8", "int8" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        CudaRunnerMode mode = CudaRunnerMode.from(cudaMode);
        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                    "CUDA runner disabled (gollek.runners.cuda.enabled=false)");
        if (mode == CudaRunnerMode.DISABLED)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                    "CUDA runner disabled (gollek.runners.cuda.mode=disabled)");
        if (mode == CudaRunnerMode.OFFLOAD)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                    "CUDA runner disabled (gollek.runners.cuda.mode=offload)");

        caps = detector.detect();
        if (!caps.available() && mode != CudaRunnerMode.FORCE) {
            throw new RunnerInitializationException(
                    ErrorCode.DEVICE_NOT_AVAILABLE.name(),
                    "CUDA unavailable on this host: " + caps.reason());
        }
        if (!caps.available()) {
            log.warnf("[CUDA] CUDA forced despite unavailable caps (%s)", caps.reason());
        }

        // Load CUDA binding library
        Path resolvedLibraryPath = resolveLibraryPath(libraryPath);
        CudaBinding.initialize(resolvedLibraryPath);
        cuda = CudaBinding.getInstance();

        if (!cuda.isNativeAvailable()) {
            log.warn("[CUDA] CUDA library not available — CPU fallback active");
            this.manifest = modelManifest;
            this.initialized = true;
            return;
        }

        // Initialize CUDA device
        int err = cuda.init(deviceId);
        if (err != 0)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                    "cudaInit() returned " + err);

        // Check for unified memory (A100/H100)
        String deviceName = caps != null ? caps.deviceName() : cuda.deviceName(deviceId);
        isUnifiedMemory = deviceName.toLowerCase().contains("a100") ||
                deviceName.toLowerCase().contains("h100") ||
                deviceName.toLowerCase().contains("h200");

        log.infof("[CUDA] Device %d: %s (unified=%s compute=%d.%d)",
                deviceId, deviceName, isUnifiedMemory,
                caps != null ? caps.cudaComputeCap() / 10 : 0,
                caps != null ? caps.cudaComputeCap() % 10 : 0);

        // Create optimization manager for auto-selecting best attention kernel
        int computeCap = caps != null ? caps.cudaComputeCap() : 0;
        boolean hasTMEM = deviceName.toLowerCase().contains("b200") || 
                         deviceName.toLowerCase().contains("b100");
        boolean supportsFP4 = computeCap >= 100;
        
        this.optimizationManager = new CudaOptimizationManager(
                computeCap, hasTMEM, supportsFP4, kvCacheManager);

        log.infof("[CUDA] Optimization: %s", optimizationManager.getRecommendedKernel());

        // Create CUDA stream
        cudaStream = cuda.streamCreate();

        // Override model dims from RunnerConfiguration
        numLayers = config.getIntParameter("num_layers", numLayers);
        numHeads = config.getIntParameter("num_heads", numHeads);
        numHeadsKv = config.getIntParameter("num_heads_kv", numHeadsKv);
        headDim = config.getIntParameter("head_dim", headDim);
        modelDim = config.getIntParameter("model_dim", modelDim);
        ffnDim = config.getIntParameter("ffn_dim", ffnDim);
        vocabSize = config.getIntParameter("vocab_size", vocabSize);

        // Allocate device memory
        long elemBytes = 2L; // FP16
        if (isUnifiedMemory) {
            // A100/H100: managed memory, zero-copy
            dResidual = cuda.mallocManaged((long) modelDim * elemBytes, 1); // CUDA_MEM_ATTACH_GLOBAL
            dNormed = cuda.mallocManaged((long) modelDim * elemBytes, 1);
            dQkv = cuda.mallocManaged((long) (numHeads + 2 * numHeadsKv) * headDim * elemBytes, 1);
            dAttnOut = cuda.mallocManaged((long) numHeads * headDim * elemBytes, 1);
            dFfnBuf = cuda.mallocManaged((long) ffnDim * 2 * elemBytes, 1);
        } else {
            // Discrete GPU: device memory
            dResidual = cuda.malloc((long) modelDim * elemBytes);
            dNormed = cuda.malloc((long) modelDim * elemBytes);
            dQkv = cuda.malloc((long) (numHeads + 2 * numHeadsKv) * headDim * elemBytes);
            dAttnOut = cuda.malloc((long) numHeads * headDim * elemBytes);
            dFfnBuf = cuda.malloc((long) ffnDim * 2 * elemBytes);
        }

        // Memory-map model weights
        weightsArena = Arena.ofAuto();
        Path modelPath = resolveModelPath(modelManifest);
        weightsMem = mmapWeights(modelPath, weightsArena);
        layerSlices = sliceLayerWeights(weightsMem, numLayers);

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[CUDA] Ready — model=%s layers=%d heads=%d/%d dim=%d ffn=%d vocab=%d",
                modelManifest.modelId(), numLayers, numHeads, numHeadsKv,
                headDim, ffnDim, vocabSize);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode using CUDA kernels in-process.
     *
     * <p>Each transformer layer runs:
     * <ol>
     *   <li>RMS Norm — {@link CudaBinding#rmsNorm}</li>
     *   <li>QKV projection GEMM — {@link CudaBinding#matmul}</li>
     *   <li>FlashAttention — {@link CudaBinding#flashAttnV2} or {@link CudaBinding#flashAttnV3}</li>
     *   <li>Output projection + residual</li>
     *   <li>SiLU-gated FFN — {@link CudaBinding#siluFfn}</li>
     * </ol>
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "CUDA runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTok = getMaxTokens(request);
        int[] prompt = tokenize(request);
        int promptLen = prompt.length;
        totalRequests.incrementAndGet();

        if (!cuda.isNativeAvailable()) {
            // CPU fallback
            float[] logits = new float[vocabSize];
            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(detokenize(sampleGreedy(logits)))
                    .model(manifest.modelId())
                    .durationMs(System.currentTimeMillis() - t0)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("fallback", true)
                    .build();
        }

        kvCacheManager.allocateForPrefill(reqId, promptLen);
        try {
            StringBuilder sb = new StringBuilder();
            int[] bt = blockTable(reqId);
            int seqLen = promptLen;

            // Prefill
            runForwardPass(seqLen, bt, false);

            // Decode loop
            for (int step = 0; step < maxTok; step++) {
                float[] logits = runForwardPass(seqLen, bt, true);
                int next = sampleGreedy(logits);
                if (isEos(next))
                    break;
                sb.append(detokenize(next));
                seqLen++;
                if (kvCacheManager.appendToken(reqId))
                    bt = blockTable(reqId);
            }

            long dur = System.currentTimeMillis() - t0;
            totalLatencyMs.addAndGet(dur);

            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(sb.toString())
                    .model(manifest.modelId())
                    .durationMs(dur)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("device", caps != null ? caps.deviceName() : "N/A")
                    .metadata("unified_mem", isUnifiedMemory)
                    .metadata("prompt_tokens", promptLen)
                    .metadata("output_tokens", sb.length())
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[CUDA] " + e.getMessage(), e);
        } finally {
            kvCacheManager.freeRequest(reqId);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            String reqId = request.getRequestId();
            int maxTok = getMaxTokens(request);
            int seqLen = tokenize(request).length;
            int seq = 0;

            if (!cuda.isNativeAvailable()) {
                float[] logits = new float[vocabSize];
                emitter.emit(StreamingInferenceChunk.finalChunk(
                        reqId, seq, detokenize(sampleGreedy(logits))));
                emitter.complete();
                return;
            }

            kvCacheManager.allocateForPrefill(reqId, seqLen);
            try {
                int[] bt = blockTable(reqId);
                runForwardPass(seqLen, bt, false);

                for (int step = 0; step < maxTok; step++) {
                    float[] logits = runForwardPass(seqLen, bt, true);
                    int next = sampleGreedy(logits);
                    boolean fin = isEos(next) || step == maxTok - 1;
                    emitter.emit(new StreamingInferenceChunk(
                            reqId, seq++, ModalityType.TEXT, detokenize(next), null, fin, fin ? "stop" : null, null, java.time.Instant.now(), null));
                    if (fin)
                        break;
                    seqLen++;
                    if (kvCacheManager.appendToken(reqId))
                        bt = blockTable(reqId);
                }
                emitter.complete();
            } catch (Exception e) {
                emitter.fail(e);
            } finally {
                kvCacheManager.freeRequest(reqId);
            }
        });
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /**
     * One forward pass through all {@code numLayers} transformer layers.
     *
     * @param seqLen     current context length
     * @param bt         block table for paged KV access
     * @param decodeOnly true = single-token decode; false = full prefill
     * @return output logits (vocabSize floats)
     */
    private float[] runForwardPass(int seqLen, int[] bt, boolean decodeOnly) {
        int T = decodeOnly ? 1 : seqLen;
        int blockSz = kvCacheManager.getConfig().getBlockSize();
        float attnScale = (float) (1.0 / Math.sqrt(headDim));
        int warpSize = 32; // NVIDIA warp size

        // KV pool slabs
        MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();

        int computeCap = caps != null ? caps.cudaComputeCap() : 0;
        boolean useFA2 = computeCap >= 80;
        boolean useFA3 = computeCap >= 90;

        for (int layer = 0; layer < numLayers; layer++) {
            MemorySegment w = layerSlices[layer];

            // Weight slice layout per layer
            long off = 0L;
            int qkvDim = (numHeads + 2 * numHeadsKv) * headDim;
            int qDim = numHeads * headDim;

            MemorySegment normW = w.asSlice(off, (long) modelDim * 4L);
            off += (long) modelDim * 4L;
            MemorySegment wQkv = w.asSlice(off, (long) modelDim * qkvDim * 4L);
            off += (long) modelDim * qkvDim * 4L;
            MemorySegment wO = w.asSlice(off, (long) qDim * modelDim * 4L);
            off += (long) qDim * modelDim * 4L;
            MemorySegment fnormW = w.asSlice(off, (long) modelDim * 4L);
            off += (long) modelDim * 4L;
            MemorySegment wGate = w.asSlice(off, (long) modelDim * ffnDim * 4L);
            off += (long) modelDim * ffnDim * 4L;
            MemorySegment wUp = w.asSlice(off, (long) modelDim * ffnDim * 4L);
            off += (long) modelDim * ffnDim * 4L;
            MemorySegment wDown = w.asSlice(off, (long) ffnDim * modelDim * 4L);

            // ── 1. Pre-attention RMS Norm ────────────────────────────────────
            cuda.rmsNorm(dNormed, dResidual, normW, modelDim, 1e-6f);

            // ── 2. QKV projection via cuBLAS ─────────────────────────────────
            cuda.matmul(dQkv, dNormed, wQkv, T, modelDim, qkvDim, 1.0f, 0.0f);

            // ── 3. Optimized attention via CudaOptimizationManager ───────────
            boolean useFp8 = optimizationManager != null && optimizationManager.shouldUseFP8();
            if (optimizationManager != null) {
                optimizationManager.executeAttention(
                        dAttnOut, dQkv, kPool, vPool,
                        1, T, numHeads, numHeadsKv, headDim,
                        attnScale, true, useFp8);
            } else if (useFA3) {
                // FlashAttention-3 with FP8 on H100+
                cuda.flashAttnV3(dAttnOut, dQkv, kPool, vPool,
                        1, T, seqLen, numHeads, headDim,
                        attnScale, 1, 1);
            } else if (useFA2) {
                // FlashAttention-2 on A100+
                cuda.flashAttnV2(dAttnOut, dQkv, kPool, vPool,
                        1, T, seqLen, numHeads, headDim,
                        attnScale, 1);
            } else {
                // Fallback to regular paged attention
                cuda.attention(dAttnOut, dQkv, kPool, vPool,
                        null, null, 1, T, numHeads, headDim,
                        blockSz, bt.length, attnScale, 1);
            }

            // ── 4. Output projection + residual ──────────────────────────────
            MemorySegment proj = isUnifiedMemory
                    ? cuda.mallocManaged((long) T * modelDim * 2L, 1)
                    : cuda.malloc((long) T * modelDim * 2L);
            cuda.matmul(proj, dAttnOut, wO, T, qDim, modelDim, 1.0f, 0.0f);
            addResidual(dResidual, proj, T * modelDim);
            cuda.free(proj);

            // ── 5. Pre-FFN RMS Norm ──────────────────────────────────────────
            cuda.rmsNorm(dNormed, dResidual, fnormW, modelDim, 1e-6f);

            // ── 6. FFN gate + up projections ─────────────────────────────────
            cuda.matmul(dFfnBuf, dNormed, wGate, T, modelDim, ffnDim, 1.0f, 0.0f);
            cuda.matmul(dFfnBuf, dNormed, wUp, T, modelDim, ffnDim, 1.0f, 0.0f);

            // ── 7. SiLU gate ─────────────────────────────────────────────────
            cuda.siluFfn(dFfnBuf, dFfnBuf, dFfnBuf, T * ffnDim);

            // ── 8. FFN down projection + residual ────────────────────────────
            cuda.matmul(dFfnBuf, dFfnBuf, wDown, T, ffnDim, modelDim, 1.0f, 0.0f);
            addResidual(dResidual, dFfnBuf, T * modelDim);

            // Synchronize after each layer
            cuda.streamSynchronize(cudaStream);
        }

        // Read logits from residual buffer
        return readLogits(dResidual, vocabSize);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void addResidual(MemorySegment a, MemorySegment b, int n) {
        if (isUnifiedMemory) {
            // Direct CPU access on A100/H100
            for (int i = 0; i < n; i++) {
                float aVal = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float bVal = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                a.setAtIndex(ValueLayout.JAVA_FLOAT, i, aVal + bVal);
            }
        } else {
            // Discrete GPU: copy to host, add, copy back
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment hA = arena.allocate((long) n * 4L, 4);
                MemorySegment hB = arena.allocate((long) n * 4L, 4);
                cuda.memcpyD2H(hA, a, (long) n * 4L);
                cuda.memcpyD2H(hB, b, (long) n * 4L);
                for (int i = 0; i < n; i++) {
                    float sum = hA.getAtIndex(ValueLayout.JAVA_FLOAT, i) +
                            hB.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    hA.setAtIndex(ValueLayout.JAVA_FLOAT, i, sum);
                }
                cuda.memcpyH2D(a, hA, (long) n * 4L);
            }
        }
    }

    private float[] readLogits(MemorySegment buffer, int vocabSize) {
        float[] logits = new float[vocabSize];
        if (isUnifiedMemory) {
            // Direct CPU read
            for (int i = 0; i < vocabSize; i++) {
                logits[i] = buffer.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            }
        } else {
            // D2H copy
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment hostBuf = arena.allocate((long) vocabSize * 4L, 4);
                cuda.memcpyD2H(hostBuf, buffer, (long) vocabSize * 4L);
                for (int i = 0; i < vocabSize; i++) {
                    logits[i] = hostBuf.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                }
            }
        }
        return logits;
    }

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    private Path resolveModelPath(ModelManifest m) {
        return m.artifacts().values().stream().findFirst()
                .map(l -> Path.of(l.uri()))
                .orElseThrow(() -> new IllegalArgumentException("No model artifact in manifest"));
    }

    private MemorySegment mmapWeights(Path modelPath, Arena arena) {
        try (RandomAccessFile raf = new RandomAccessFile(modelPath.toFile(), "r");
                FileChannel ch = raf.getChannel()) {
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            log.infof("[CUDA] Weights mmap'd: %s (%.1f GB)", modelPath.getFileName(), ch.size() / 1e9);
            return seg;
        } catch (Exception e) {
            log.warnf("[CUDA] Cannot mmap %s: %s — using zero weights", modelPath, e.getMessage());
            return arena.allocate(256L * 1024 * 1024, 64);
        }
    }

    private MemorySegment[] sliceLayerWeights(MemorySegment weights, int numLayers) {
        MemorySegment[] slices = new MemorySegment[numLayers];
        long layerBytes = weights.byteSize() / numLayers;
        for (int i = 0; i < numLayers; i++) {
            slices[i] = weights.asSlice((long) i * layerBytes, layerBytes);
        }
        return slices;
    }

    @Override
    public boolean health() {
        return initialized && (cuda == null || cuda.isNativeAvailable());
    }

    @Override
    public void close() {
        initialized = false;
        if (cuda != null && cuda.isNativeAvailable()) {
            if (!cudaStream.equals(MemorySegment.NULL)) {
                cuda.streamDestroy(cudaStream);
            }
            for (MemorySegment buf : List.of(dResidual, dNormed, dQkv, dAttnOut, dFfnBuf)) {
                if (buf != null && !buf.equals(MemorySegment.NULL)) {
                    cuda.free(buf);
                }
            }
            if (weightsArena != null) {
                weightsArena.close();
            }
        }
        log.info("[CUDA] Runner closed");
    }
}
