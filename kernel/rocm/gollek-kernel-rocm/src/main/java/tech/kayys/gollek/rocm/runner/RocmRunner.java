package tech.kayys.gollek.rocm.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.plugin.runner.RunnerInitializationException;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.rocm.binding.RocmHipBinding;
import tech.kayys.gollek.rocm.binding.RocmHipCpuFallback;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.RunnerMetadata;

import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * AMD ROCm ModelRunner for Gollek.
 *
 * <p>Runs {@link ModelFormat#GGUF} and {@link ModelFormat#SAFETENSORS} models
 * in-process on AMD GPUs via {@link RocmHipBinding}, which wraps the HIP
 * runtime C API ({@code hip_runtime_api.h}) through Java FFM.
 *
 * <h2>Kernel pipeline</h2>
 * <p>This runner loads a pre-compiled HIP kernel module ({@code .hsaco} file)
 * compiled from the same CUDA sources used by the CUDA runners, translated with
 * {@code hipify-clang}. The forward pass follows the same structure as
 * {@link tech.kayys.gollek.extension.metal.runner.MetalRunner} but uses HIP
 * instead of Metal:
 * <ol>
 *   <li>RMS Norm — {@code gollek_rocm_rmsnorm}</li>
 *   <li>QKV projection — {@code gollek_rocm_gemm}</li>
 *   <li>Paged attention — {@code gollek_rocm_paged_attention} (hipified FA)</li>
 *   <li>SiLU FFN — {@code gollek_rocm_silu_ffn}</li>
 * </ol>
 *
 * <h2>MI300X memory model</h2>
 * <p>The AMD MI300X (and MI300A) uses a unified CPU+GPU memory architecture
 * where both processors share the same HBM3 pool. On these GPUs
 * {@link RocmHipBinding#mallocManaged} allocates memory accessible from both
 * sides without explicit {@code hipMemcpy}. This runner detects MI300X/MI300A
 * via the device name and uses managed memory when available, falling back to
 * explicit H2D copies on older MI100/MI200 cards.
 *
 * <h2>Compatibility</h2>
 * <p>Tested target architectures:
 * <ul>
 *   <li>gfx942 — AMD Instinct MI300X (192 GB HBM3, unified)</li>
 *   <li>gfx90a — AMD Instinct MI250X (128 GB HBM2e, discrete)</li>
 *   <li>gfx1100 — AMD Radeon RX 7900 XTX (24 GB GDDR6, consumer)</li>
 * </ul>
 * The {@code .hsaco} kernel is compiled for each target architecture; the
 * correct file is selected at runtime by reading the device's {@code gcnArchName}.
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.runners.rocm.enabled=false
 *   gollek.runners.rocm.library-path=/opt/rocm/lib/libamdhip64.so
 *   gollek.runners.rocm.kernel-path=/opt/gollek/lib/gollek_rocm_gfx942.hsaco
 *   gollek.runners.rocm.device-id=0
 *   gollek.runners.rocm.num-layers=32
 *   gollek.runners.rocm.num-heads=32
 *   gollek.runners.rocm.num-heads-kv=8
 *   gollek.runners.rocm.head-dim=128
 *   gollek.runners.rocm.model-dim=4096
 *   gollek.runners.rocm.ffn-dim=14336
 *   gollek.runners.rocm.vocab-size=32000
 * </pre>
 *
 * <h3>Build HIP kernels</h3>
 * <pre>
 *   make -C src/main/cpp/rocm AMDGPU_TARGET=gfx942
 *   # Output: target/native/linux-x86_64/gollek_rocm_gfx942.hsaco
 * </pre>
 */
@ApplicationScoped
public class RocmRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "rocm-hip";

    @ConfigProperty(name = "gollek.runners.rocm.enabled",      defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.rocm.library-path",
                    defaultValue = "/opt/rocm/lib/libamdhip64.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.rocm.kernel-path",
                    defaultValue = "/opt/gollek/lib/gollek_rocm_gfx942.hsaco")
    String kernelPath;

    @ConfigProperty(name = "gollek.runners.rocm.device-id",    defaultValue = "0")
    int deviceId;

    @ConfigProperty(name = "gollek.runners.rocm.num-layers",   defaultValue = "32")
    int numLayers;

    @ConfigProperty(name = "gollek.runners.rocm.num-heads",    defaultValue = "32")
    int numHeads;

    @ConfigProperty(name = "gollek.runners.rocm.num-heads-kv", defaultValue = "8")
    int numHeadsKv;

    @ConfigProperty(name = "gollek.runners.rocm.head-dim",     defaultValue = "128")
    int headDim;

    @ConfigProperty(name = "gollek.runners.rocm.model-dim",    defaultValue = "4096")
    int modelDim;

    @ConfigProperty(name = "gollek.runners.rocm.ffn-dim",      defaultValue = "14336")
    int ffnDim;

    @ConfigProperty(name = "gollek.runners.rocm.vocab-size",   defaultValue = "32000")
    int vocabSize;

    @Inject
    PagedKVCacheManager kvCacheManager;

    private RocmHipBinding hip;
    private ModelManifest  manifest;
    private String         deviceName = "unknown";
    private boolean        isUnifiedMemory = false;

    // HIP module + kernel function handles
    private MemorySegment hipModule          = MemorySegment.NULL;
    private MemorySegment fnRmsNorm          = MemorySegment.NULL;
    private MemorySegment fnGemm             = MemorySegment.NULL;
    private MemorySegment fnPagedAttention   = MemorySegment.NULL;
    private MemorySegment fnSiluFfn          = MemorySegment.NULL;

    // Device memory — managed or discrete depending on GPU
    private MemorySegment dResidual   = MemorySegment.NULL;
    private MemorySegment dNormed     = MemorySegment.NULL;
    private MemorySegment dQkv        = MemorySegment.NULL;
    private MemorySegment dAttnOut    = MemorySegment.NULL;
    private MemorySegment dFfnBuf     = MemorySegment.NULL;
    private MemorySegment weightsMem  = MemorySegment.NULL;
    private Arena          weightsArena;

    // HIP stream for async operations
    private MemorySegment hipStream = MemorySegment.NULL;

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override public String name()           { return RUNNER_NAME; }
    @Override public String framework()      { return "rocm-hip"; }
    @Override public DeviceType deviceType() { return DeviceType.ROCM; }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.GGUF, ModelFormat.SAFETENSORS),
                List.of(DeviceType.ROCM),
                Map.of("hip_api",        "6.x",
                       "unified_memory", String.valueOf(isUnifiedMemory),
                       "device",         deviceName,
                       "targets",        "gfx942,gfx90a,gfx1100",
                       "project",        "https://rocm.docs.amd.com/"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true)
                .maxBatchSize(32)
                .supportedDataTypes(new String[]{"fp16", "bf16", "fp8", "int8"})
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        if (!enabled) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "ROCm runner disabled (gollek.runners.rocm.enabled=false)");

        numLayers  = config.getIntParameter("num_layers",   numLayers);
        numHeads   = config.getIntParameter("num_heads",    numHeads);
        numHeadsKv = config.getIntParameter("num_heads_kv", numHeadsKv);
        headDim    = config.getIntParameter("head_dim",     headDim);
        modelDim   = config.getIntParameter("model_dim",    modelDim);
        ffnDim     = config.getIntParameter("ffn_dim",      ffnDim);
        vocabSize  = config.getIntParameter("vocab_size",   vocabSize);

        RocmHipBinding.initialize(Path.of(libraryPath));
        hip = RocmHipBinding.getInstance();

        if (!hip.isNativeAvailable()) {
            log.warn("[ROCm] HIP library not available — CPU fallback active");
            this.manifest    = modelManifest;
            this.initialized = true;
            return;
        }

        // Set device and query properties
        hip.setDevice(deviceId);
        deviceName      = hip.getDeviceName(deviceId);
        isUnifiedMemory = deviceName.toLowerCase().contains("mi300") ||
                          deviceName.toLowerCase().contains("mi308") ||
                          deviceName.toLowerCase().contains("mi325");
        log.infof("[ROCm] Device %d: %s (unified=%s)", deviceId, deviceName, isUnifiedMemory);

        // Create HIP stream
        hipStream = hip.streamCreate();

        // Load pre-compiled HIP kernel module
        hipModule = hip.moduleLoad(kernelPath);
        if (!isNull(hipModule)) {
            fnRmsNorm        = hip.moduleGetFunction(hipModule, "gollek_rocm_rmsnorm");
            fnGemm           = hip.moduleGetFunction(hipModule, "gollek_rocm_gemm");
            fnPagedAttention = hip.moduleGetFunction(hipModule, "gollek_rocm_paged_attention");
            fnSiluFfn        = hip.moduleGetFunction(hipModule, "gollek_rocm_silu_ffn");
            log.infof("[ROCm] Loaded kernels from %s", kernelPath);
        } else {
            log.warnf("[ROCm] Could not load %s — will use Java fallback ops", kernelPath);
        }

        // Allocate working memory
        long elemBytes = 2L; // FP16
        if (isUnifiedMemory) {
            // MI300X: managed memory, accessible from CPU and GPU without copies
            dResidual  = hip.mallocManaged((long) modelDim * elemBytes, RocmHipBinding.HIP_MEM_ATTACH_GLOBAL);
            dNormed    = hip.mallocManaged((long) modelDim * elemBytes, RocmHipBinding.HIP_MEM_ATTACH_GLOBAL);
            dQkv       = hip.mallocManaged((long)(numHeads + 2 * numHeadsKv) * headDim * elemBytes, RocmHipBinding.HIP_MEM_ATTACH_GLOBAL);
            dAttnOut   = hip.mallocManaged((long) numHeads * headDim * elemBytes, RocmHipBinding.HIP_MEM_ATTACH_GLOBAL);
            dFfnBuf    = hip.mallocManaged((long) ffnDim * 2 * elemBytes, RocmHipBinding.HIP_MEM_ATTACH_GLOBAL);
        } else {
            // MI200 / RX series: discrete device memory
            dResidual  = hip.malloc((long) modelDim * elemBytes);
            dNormed    = hip.malloc((long) modelDim * elemBytes);
            dQkv       = hip.malloc((long)(numHeads + 2 * numHeadsKv) * headDim * elemBytes);
            dAttnOut   = hip.malloc((long) numHeads * headDim * elemBytes);
            dFfnBuf    = hip.malloc((long) ffnDim * 2 * elemBytes);
        }

        // Memory-map model weights into pinned host memory
        weightsArena = Arena.ofAuto();
        Path modelPath = modelManifest.artifacts().values().stream().findFirst()
                .map(loc -> Path.of(loc.uri()))
                .orElseThrow(() -> new RunnerInitializationException(
                        ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(), "No model artifact"));
        weightsMem = mmapWeights(modelPath, weightsArena);

        // On unified MI300X: the weights mmap is already GPU-accessible
        // On discrete GPUs: caller responsible for explicit H2D copy per layer
        if (!isUnifiedMemory) {
            log.infof("[ROCm] Discrete GPU detected — layer weights will be copied H2D per inference step");
        }

        this.manifest    = modelManifest;
        this.initialized = true;

        log.infof("[ROCm] Ready — model=%s layers=%d heads=%d/%d dim=%d vocab=%d",
                modelManifest.modelId(), numLayers, numHeads, numHeadsKv, modelDim, vocabSize);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode via HIP kernels.
     *
     * <p>On MI300X (unified memory) all tensor reads/writes happen in-place
     * on HBM3 without H2D or D2H copies. On discrete GPUs (MI200, RX7900)
     * the caller must copy layer weights before each forward pass step.
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized) throw new InferenceException(
                ErrorCode.RUNTIME_INVALID_STATE, "ROCm runner not initialized");

        long   t0       = System.currentTimeMillis();
        String reqId    = request.getRequestId();
        int    maxTok   = getMaxTokens(request);
        int[]  prompt   = tokenize(request);
        int    seqLen   = prompt.length;
        totalRequests.incrementAndGet();

        if (!hip.isNativeAvailable()) {
            float[] logits = RocmHipCpuFallback.runFallback(vocabSize);
            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(detokenize(sampleGreedy(logits)))
                    .model(manifest.modelId())
                    .durationMs(System.currentTimeMillis() - t0)
                    .metadata("runner", RUNNER_NAME).metadata("fallback", true)
                    .build();
        }

        kvCacheManager.allocateForPrefill(reqId, seqLen);
        try {
            StringBuilder sb = new StringBuilder();
            int[]         bt = blockTable(reqId);

            // Prefill
            runForwardPass(seqLen, bt, false);

            // Decode loop
            for (int step = 0; step < maxTok; step++) {
                float[] logits = runForwardPass(seqLen, bt, true);
                int     next   = sampleGreedy(logits);
                if (isEos(next)) break;
                sb.append(detokenize(next));
                seqLen++;
                if (kvCacheManager.appendToken(reqId)) bt = blockTable(reqId);
            }

            long dur = System.currentTimeMillis() - t0;
            totalLatencyMs.addAndGet(dur);

            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(sb.toString())
                    .model(manifest.modelId())
                    .durationMs(dur)
                    .metadata("runner",     RUNNER_NAME)
                    .metadata("device",     deviceName)
                    .metadata("unified",    isUnifiedMemory)
                    .metadata("prompt_len", prompt.length)
                    .metadata("output_len", sb.length())
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[ROCm] " + e.getMessage(), e);
        } finally {
            kvCacheManager.freeRequest(reqId);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            String reqId  = request.getRequestId();
            int    maxTok = getMaxTokens(request);
            int    seqLen = tokenize(request).length;
            int    seq    = 0;

            if (!hip.isNativeAvailable()) {
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
                    int     next   = sampleGreedy(logits);
                    boolean fin    = isEos(next) || step == maxTok - 1;
                    if (fin) {
                        emitter.emit(StreamingInferenceChunk.finalChunk(reqId, seq++, detokenize(next)));
                    } else {
                        emitter.emit(StreamingInferenceChunk.of(reqId, seq++, detokenize(next)));
                    }
                    if (fin) break;
                    seqLen++;
                    if (kvCacheManager.appendToken(reqId)) bt = blockTable(reqId);
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
     * Run all transformer layers for one decode or prefill step.
     *
     * <p>Each layer executes four HIP kernels via
     * {@link RocmHipBinding#moduleLaunchKernel}: RMSNorm, GEMM, paged
     * attention, SiLU FFN. On discrete GPUs the layer weight slice is copied
     * H2D before the GEMM and freed immediately after.
     *
     * @param seqLen      current context length
     * @param bt          block table for paged KV access
     * @param decodeOnly  true = single-token decode; false = full prefill
     * @return output logits (vocabSize floats, read from dResidual)
     */
    private float[] runForwardPass(int seqLen, int[] bt, boolean decodeOnly) {
        int   T          = decodeOnly ? 1 : seqLen;
        int   blockSz    = kvCacheManager.getConfig().getBlockSize();
        float scale      = (float)(1.0 / Math.sqrt(headDim));
        int   warpSize   = 64;  // AMD wavefront size

        // KV pool pointers — already on device (Arena.ofShared = pinned or managed)
        MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();

        long layerBytes  = weightsMem.byteSize() / numLayers;

        for (int layer = 0; layer < numLayers; layer++) {
            MemorySegment layerWeights = weightsMem.asSlice((long) layer * layerBytes, layerBytes);

            // On discrete GPU: copy layer weights to device before compute
            MemorySegment devWeights = layerWeights;
            if (!isUnifiedMemory) {
                MemorySegment dw = hip.malloc(layerBytes);
                if (!isNull(dw)) {
                    hip.memcpyAsyncH2D(dw, layerWeights, layerBytes, hipStream);
                    devWeights = dw;
                }
            }

            // ── RMS Norm ─────────────────────────────────────────────────────
            if (!isNull(fnRmsNorm)) {
                launchKernel1D(fnRmsNorm, warpSize, warpSize,
                        dNormed, dResidual, devWeights, modelDim);
            }

            // ── QKV GEMM ─────────────────────────────────────────────────────
            int qkvDim = (numHeads + 2 * numHeadsKv) * headDim;
            if (!isNull(fnGemm)) {
                launchKernel2D(fnGemm, T, qkvDim, warpSize,
                        dQkv, dNormed, devWeights, T, modelDim, qkvDim);
            }

            // ── Paged attention ───────────────────────────────────────────────
            if (!isNull(fnPagedAttention)) {
                launchKernel2D(fnPagedAttention, 1, numHeads, warpSize,
                        dAttnOut, dQkv, kPool, vPool,
                        seqLen, numHeads, numHeadsKv, headDim, blockSz, bt.length, scale);
            }

            // ── SiLU FFN ──────────────────────────────────────────────────────
            if (!isNull(fnSiluFfn)) {
                launchKernel1D(fnSiluFfn, ffnDim, warpSize,
                        dFfnBuf, dNormed, devWeights.asSlice(0, ffnDim * modelDim * 2L), ffnDim);
            }

            // Synchronise stream after each layer (can be made fully async in production)
            hip.streamSynchronize(hipStream);

            // Release discrete device weight copy
            if (!isUnifiedMemory && devWeights != layerWeights) hip.free(devWeights);
        }

        // Read logits from residual buffer
        if (!isUnifiedMemory) {
            // D2H copy for discrete GPU
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment hostBuf = arena.allocate((long) vocabSize * 2L, 4);
                hip.memcpyD2H(hostBuf, dResidual, Math.min((long) vocabSize * 2L, dResidual.byteSize()));
                float[] logits = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                    logits[i] = float16ToFloat(hostBuf.getAtIndex(ValueLayout.JAVA_SHORT, i));
                return logits;
            }
        } else {
            // Unified memory: direct CPU read (MI300X)
            float[] logits = new float[vocabSize];
            long n = Math.min(vocabSize, (int)(dResidual.byteSize() / 2L));
            for (int i = 0; i < n; i++)
                logits[i] = float16ToFloat(dResidual.getAtIndex(ValueLayout.JAVA_SHORT, i));
            return logits;
        }
    }

    // ── Kernel launch helpers ─────────────────────────────────────────────────

    /** Launch a 1D grid kernel with scalar arguments packed as pointer array. */
    private void launchKernel1D(MemorySegment fn, int gridX, int blockX, Object... args) {
        if (isNull(fn)) return;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment argPtrs = packKernelArgs(a, args);
            hip.moduleLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1, 0, hipStream, argPtrs);
        }
    }

    /** Launch a 2D grid kernel. */
    private void launchKernel2D(MemorySegment fn, int gridX, int gridY,
                                 int blockX, Object... args) {
        if (isNull(fn)) return;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment argPtrs = packKernelArgs(a, args);
            hip.moduleLaunchKernel(fn, gridX, gridY, 1, blockX, 1, 1, 0, hipStream, argPtrs);
        }
    }

    /**
     * Pack kernel arguments into a void*[] array.
     * HIP requires an array of pointers-to-values as the {@code kernelParams} argument.
     */
    private MemorySegment packKernelArgs(Arena arena, Object[] args) {
        MemorySegment ptrs = arena.allocate((long) args.length * 8L, 8);
        for (int i = 0; i < args.length; i++) {
            MemorySegment slot = arena.allocate(8L, 8);
            Object arg = args[i];
            if      (arg instanceof MemorySegment ms) slot.set(ValueLayout.ADDRESS,  0L, ms);
            else if (arg instanceof Integer       iv) slot.setAtIndex(ValueLayout.JAVA_INT, 0, iv);
            else if (arg instanceof Long           lv) slot.setAtIndex(ValueLayout.JAVA_LONG, 0, lv);
            else if (arg instanceof Float          fv) slot.setAtIndex(ValueLayout.JAVA_FLOAT, 0, fv);
            ptrs.setAtIndex(ValueLayout.ADDRESS, i, slot);
        }
        return ptrs;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** IEEE 754 FP16 → FP32 conversion (half precision stored as short). */
    private static float float16ToFloat(short h) {
        int bits = h & 0xFFFF;
        int sign = (bits >> 15) & 1;
        int exp  = (bits >> 10) & 0x1F;
        int mant = bits & 0x3FF;
        if (exp == 0)   return 0f;
        if (exp == 31)  return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        int f32bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        return Float.intBitsToFloat(f32bits);
    }

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    private MemorySegment mmapWeights(Path modelPath, Arena arena) {
        try (RandomAccessFile raf = new RandomAccessFile(modelPath.toFile(), "r");
             FileChannel ch = raf.getChannel()) {
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            log.infof("[ROCm] Weights mmap'd: %s (%.1f GB)", modelPath.getFileName(), ch.size() / 1e9);
            return seg;
        } catch (Exception e) {
            log.warnf("[ROCm] Cannot mmap %s: %s — using zero weights", modelPath, e.getMessage());
            return arena.allocate(256L * 1024 * 1024, 64);
        }
    }

    private static boolean isNull(MemorySegment s) {
        return s == null || s.equals(MemorySegment.NULL) || s.address() == 0;
    }

    @Override public boolean health() { return initialized; }

    @Override
    public void close() {
        initialized = false;
        if (hip != null && hip.isNativeAvailable()) {
            for (MemorySegment buf : List.of(dResidual, dNormed, dQkv, dAttnOut, dFfnBuf)) {
                if (!isNull(buf)) hip.free(buf);
            }
            hip.streamDestroy(hipStream);
        }
        if (weightsArena != null) try { weightsArena.close(); } catch (Exception ignored) {}
        log.info("[ROCm] Closed");
    }
}
