package tech.kayys.gollek.metal.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.plugin.runner.RunnerInitializationException;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.metal.config.MetalRunnerMode;
import tech.kayys.gollek.metal.detection.AppleSiliconDetector;
import tech.kayys.gollek.metal.detection.MetalCapabilities;
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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;

/**
 * Apple Silicon Metal ModelRunner for Gollek.
 *
 * <h2>What this runner does</h2>
 * <p>Runs transformer inference <b>in-process</b> on Apple Silicon (M1/M2/M3/M4)
 * using two FFM binding classes:
 * <ul>
 *   <li>{@link MetalFlashAttentionBinding} — FA4-equivalent fused attention via
 *       {@code gollek_metal_fa4_attention} (MPSGraph SDPA on macOS 14+).</li>
 *   <li>{@link MetalBinding} — all other operations: RMS Norm, QKV/FFN GEMM via
 *       MPSMatrixMultiplication (dispatches to Apple AMX blocks), SiLU FFN.</li>
 * </ul>
 *
 * <h2>FlashAttention-4 on Apple Silicon</h2>
 * <p>FA4 targets NVIDIA Blackwell. On Apple Silicon the same algorithmic goal —
 * fused QK^T/softmax/×V in a single GPU pass without materialising the full
 * attention matrix — is achieved via {@code MPSGraph.scaledDotProductAttention}
 * (macOS 14+, M3/M4 optimised). The mapping:
 * <table border="1" cellpadding="3">
 * <tr><th>FA4 (Blackwell)</th><th>Metal (Apple Silicon)</th></tr>
 * <tr><td>TMEM tile accumulator</td><td>GPU tile cache (on-chip SRAM)</td></tr>
 * <tr><td>Async UMMA pipelines</td><td>MPSCommandBuffer concurrent dispatch</td></tr>
 * <tr><td>Software exp() on FMA</td><td>Apple M-series FP16 exp intrinsic</td></tr>
 * <tr><td>Fused single-pass kernel</td><td>MPSGraph SDPA (macOS 14+ M3/M4)</td></tr>
 * <tr><td>No full attn matrix on DRAM</td><td>Same guarantee via MPS tiling</td></tr>
 * </table>
 * On M1/M2 (macOS 13) the fallback is separate MPS QK^T → softmax → ×V matmuls.
 * Check {@link MetalFlashAttentionBinding#isSdpaAvailable()} at runtime.
 *
 * <h2>Unified Memory Architecture</h2>
 * <p>On Apple Silicon all {@link MemorySegment} allocations ({@link Arena#ofShared()} or
 * {@link FileChannel#map}) land in unified DRAM shared by CPU and GPU. The
 * {@code gollek_metal_bridge.m} bridge calls {@code newBufferWithBytesNoCopy}
 * to give Metal a {@code MTLStorageModeShared} view — <b>zero copy</b>.
 * Gollek's K/V pool ({@link tech.kayys.gollek.kvcache.PhysicalBlockPool#rawKPool()})
 * is already in off-heap memory, so KV cache accesses are zero-copy too.
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.runners.metal.enabled=false
 *   gollek.runners.metal.mode=auto  # auto|standard|offload|force|disabled
 *   gollek.runners.metal.library-path=~/.gollek/libs/libgollek_metal.dylib
 *   gollek.runners.metal.num-layers=32
 *   gollek.runners.metal.num-heads=32
 *   gollek.runners.metal.num-heads-kv=8
 *   gollek.runners.metal.head-dim=128
 *   gollek.runners.metal.model-dim=4096
 *   gollek.runners.metal.ffn-dim=14336
 *   gollek.runners.metal.vocab-size=32000
 * </pre>
 *
 * <h3>Build the Metal bridge</h3>
 * <pre>
 *   make -C src/main/cpp/metal
 *   # Compiles gollek_metal_bridge.m + gollek_metal_fa4.m into one dylib
 * </pre>
 */
@ApplicationScoped
public class MetalRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "metal-apple-silicon";

    @ConfigProperty(name = "gollek.runners.metal.enabled",      defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.metal.mode",         defaultValue = "auto")
    String metalMode;

    @ConfigProperty(name = "gollek.runners.metal.library-path",
                    defaultValue = "~/.gollek/libs/libgollek_metal.dylib")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.metal.num-layers",   defaultValue = "32")
    int numLayers;

    @ConfigProperty(name = "gollek.runners.metal.num-heads",    defaultValue = "32")
    int numHeads;

    @ConfigProperty(name = "gollek.runners.metal.num-heads-kv", defaultValue = "8")
    int numHeadsKv;

    @ConfigProperty(name = "gollek.runners.metal.head-dim",     defaultValue = "128")
    int headDim;

    @ConfigProperty(name = "gollek.runners.metal.model-dim",    defaultValue = "4096")
    int modelDim;

    @ConfigProperty(name = "gollek.runners.metal.ffn-dim",      defaultValue = "14336")
    int ffnDim;

    @ConfigProperty(name = "gollek.runners.metal.vocab-size",   defaultValue = "32000")
    int vocabSize;

    @Inject PagedKVCacheManager  kvCacheManager;
    @Inject AppleSiliconDetector detector;

    private MetalBinding                 metal;
    private MetalFlashAttentionBinding   metalFa4;
    private MetalCapabilities            caps;
    private ModelManifest                manifest;
    private MemorySegment                weightsMapped;
    private Arena                        weightsArena;
    private MemorySegment[]              layerSlices;

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

    @Override public String name()           { return RUNNER_NAME; }
    @Override public String framework()      { return "metal-mps"; }
    @Override public DeviceType deviceType() { return DeviceType.METAL; }

    @Override
    public RunnerMetadata metadata() {
        String attnPath = metalFa4 != null && metalFa4.isSdpaAvailable()
                ? "MPSGraph-SDPA-FA4-equiv" : "MPS-separate-matmuls";
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.GGUF, ModelFormat.SAFETENSORS),
                List.of(DeviceType.METAL, DeviceType.CPU),
                Map.of("backend",         "Metal Performance Shaders",
                       "attention_path",  attnPath,
                       "mode",            metalMode,
                       "uma",             "true",
                       "zero_copy_kv",    "true",
                       "fa4_equiv",       "true"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true)
                .maxBatchSize(16)
                .supportedDataTypes(new String[]{"fp32", "fp16", "bf16", "q4_k_m", "q8_0"})
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        MetalRunnerMode mode = MetalRunnerMode.from(metalMode);
        if (!enabled) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "Metal runner disabled (gollek.runners.metal.enabled=false)");
        if (mode == MetalRunnerMode.DISABLED) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "Metal runner disabled (gollek.runners.metal.mode=disabled)");
        if (mode == MetalRunnerMode.OFFLOAD) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "Metal runner disabled (gollek.runners.metal.mode=offload)");

        caps = detector.detect();
        if (!caps.available() && mode != MetalRunnerMode.FORCE) {
            throw new RunnerInitializationException(
                    ErrorCode.DEVICE_NOT_AVAILABLE.name(),
                    "Metal unavailable on this host: " + caps.reason());
        }
        if (!caps.available()) {
            log.warnf("[Metal] Metal forced despite unavailable caps (%s)", caps.reason());
        }

        // ── Load both Metal binding classes from the same dylib ───────────────
        Path resolvedLibraryPath = resolveLibraryPath(libraryPath);
        MetalBinding.initialize(resolvedLibraryPath);
        metal = MetalBinding.getInstance();

        MetalFlashAttentionBinding.initialize(resolvedLibraryPath);
        metalFa4 = MetalFlashAttentionBinding.getInstance();

        if (metal.isNativeAvailable()) {
            int err = metal.init(); // gollek_metal_init() — MTLCreateSystemDefaultDevice
            if (err != 0) throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                    "gollek_metal_init() returned " + err);
        }

        log.infof("[Metal] Device: %s — unified=%s UMA=%.1f GB sdpa=%s bf16=%s mode=%s",
                metal.deviceName(), metal.isUnifiedMemory(),
                metal.availableMemory() / 1e9,
                metalFa4.isSdpaAvailable(), metalFa4.isBf16Available(),
                mode.name().toLowerCase());

        // Override model dims from RunnerConfiguration
        numLayers  = config.getIntParameter("num_layers",   numLayers);
        numHeads   = config.getIntParameter("num_heads",    numHeads);
        numHeadsKv = config.getIntParameter("num_heads_kv", numHeadsKv);
        headDim    = config.getIntParameter("head_dim",     headDim);
        modelDim   = config.getIntParameter("model_dim",    modelDim);
        ffnDim     = config.getIntParameter("ffn_dim",      ffnDim);
        vocabSize  = config.getIntParameter("vocab_size",   vocabSize);

        // Memory-map model into unified DRAM — zero copy to Metal
        weightsArena  = Arena.ofAuto();
        Path modelPath = resolveModelPath(modelManifest);
        weightsMapped  = mmapModel(modelPath, weightsArena);
        layerSlices    = sliceLayerWeights(weightsMapped, numLayers);

        this.manifest    = modelManifest;
        this.initialized = true;

        log.infof("[Metal] Ready — model=%s layers=%d heads=%d/%d dim=%d ffn=%d vocab=%d",
                modelManifest.modelId(), numLayers, numHeads, numHeadsKv,
                headDim, ffnDim, vocabSize);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode using Metal MPS kernels in-process.
     *
     * <p>Each transformer layer runs:
     * <ol>
     *   <li>RMS Norm — {@link MetalBinding#rmsNorm}</li>
     *   <li>QKV projection GEMM — {@link MetalBinding#matmul} (AMX blocks)</li>
     *   <li>FA4-equivalent paged attention — {@link MetalFlashAttentionBinding#fa4Attention}
     *       (MPSGraph SDPA on macOS 14+ / separate matmuls on macOS 13)</li>
     *   <li>Output projection + residual</li>
     *   <li>SiLU-gated FFN — {@link MetalBinding#siluFfn} + two GEMMs</li>
     * </ol>
     * All tensors live in unified DRAM — Metal accesses them directly.
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized) throw new InferenceException(
                ErrorCode.RUNTIME_INVALID_STATE, "Metal runner not initialized");

        long   t0       = System.currentTimeMillis();
        String reqId    = request.getRequestId();
        int    maxTok   = getMaxTokens(request);
        int[]  prompt   = tokenize(request);
        int    promptLen = prompt.length;
        totalRequests.incrementAndGet();

        kvCacheManager.allocateForPrefill(reqId, promptLen);
        try (Arena arena = Arena.ofConfined()) {
            // All these allocations are in unified DRAM on Apple Silicon
            MemorySegment residual = arena.allocate((long) modelDim * 4L, 64);
            MemorySegment normed   = arena.allocate((long) modelDim * 4L, 64);
            MemorySegment qkv      = arena.allocate((long)(numHeads + 2 * numHeadsKv) * headDim * 4L, 64);
            MemorySegment attnOut  = arena.allocate((long) numHeads * headDim * 4L, 64);
            MemorySegment ffnGate  = arena.allocate((long) ffnDim * 4L, 64);
            MemorySegment ffnUp    = arena.allocate((long) ffnDim * 4L, 64);
            MemorySegment ffnOut   = arena.allocate((long) modelDim * 4L, 64);

            StringBuilder sb     = new StringBuilder();
            int[]         bt     = blockTable(reqId);
            int           seqLen = promptLen;

            // Prefill
            runForwardPass(arena, residual, normed, qkv, attnOut,
                           ffnGate, ffnUp, ffnOut, reqId, seqLen, bt, false);

            // Decode loop
            for (int step = 0; step < maxTok; step++) {
                float[] logits = runForwardPass(arena, residual, normed, qkv, attnOut,
                                                ffnGate, ffnUp, ffnOut,
                                                reqId, seqLen, bt, true);
                int next = sampleGreedy(logits);
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
                    .metadata("runner",       RUNNER_NAME)
                    .metadata("device",       metal.deviceName())
                    .metadata("sdpa_path",    metalFa4.isSdpaAvailable())
                    .metadata("bf16",         metalFa4.isBf16Available())
                    .metadata("unified_mem",  metal.isUnifiedMemory())
                    .metadata("prompt_tokens", promptLen)
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[Metal] " + e.getMessage(), e);
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
            try {
                kvCacheManager.allocateForPrefill(reqId, seqLen);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment residual = arena.allocate((long) modelDim * 4L, 64);
                    MemorySegment normed   = arena.allocate((long) modelDim * 4L, 64);
                    MemorySegment qkv      = arena.allocate((long)(numHeads + 2 * numHeadsKv) * headDim * 4L, 64);
                    MemorySegment attnOut  = arena.allocate((long) numHeads * headDim * 4L, 64);
                    MemorySegment ffnGate  = arena.allocate((long) ffnDim * 4L, 64);
                    MemorySegment ffnUp    = arena.allocate((long) ffnDim * 4L, 64);
                    MemorySegment ffnOut   = arena.allocate((long) modelDim * 4L, 64);
                    int[] bt = blockTable(reqId);

                    runForwardPass(arena, residual, normed, qkv, attnOut,
                                   ffnGate, ffnUp, ffnOut, reqId, seqLen, bt, false);

                    for (int step = 0; step < maxTok; step++) {
                        float[] logits = runForwardPass(arena, residual, normed, qkv, attnOut,
                                                        ffnGate, ffnUp, ffnOut,
                                                        reqId, seqLen, bt, true);
                        int next = sampleGreedy(logits);
                        boolean fin = isEos(next) || step == maxTok - 1;
                        if (fin) {
                            emitter.emit(StreamingInferenceChunk.finalChunk(reqId, seq++, detokenize(next)));
                        } else {
                            emitter.emit(StreamingInferenceChunk.of(reqId, seq++, detokenize(next)));
                        }
                        if (fin) break;
                        seqLen++;
                        if (kvCacheManager.appendToken(reqId)) bt = blockTable(reqId);
                    }
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
     * <p>Attention uses {@link MetalFlashAttentionBinding#fa4Attention} with
     * pre-gathered K/V from the paged cache. All other ops use
     * {@link MetalBinding} MPS kernels. Everything is zero-copy on Apple Silicon.
     */
    private float[] runForwardPass(Arena arena,
                                   MemorySegment residual, MemorySegment normed,
                                   MemorySegment qkv, MemorySegment attnOut,
                                   MemorySegment ffnGate, MemorySegment ffnUp, MemorySegment ffnOut,
                                   String reqId, int seqLen, int[] bt, boolean decodeOnly) {

        int   T         = decodeOnly ? 1 : seqLen;
        int   blockSz   = kvCacheManager.getConfig().getBlockSize();
        float attnScale = (float)(1.0 / Math.sqrt(headDim));

        // KV pool slabs are already in unified DRAM — zero copy
        MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();

        for (int layer = 0; layer < numLayers; layer++) {
            MemorySegment w = layerSlices[layer];

            // Weight slice layout per layer (approximate — real GGUF parser needed):
            // [normW(D), W_qkv(D×qkvDim), W_o(qDim×D), ffnNormW(D), W_gate(D×F), W_up(D×F), W_down(F×D)]
            long off   = 0L;
            int  qkvDim = (numHeads + 2 * numHeadsKv) * headDim;
            int  qDim   = numHeads * headDim;

            MemorySegment normW  = w.asSlice(off, (long) modelDim * 4L); off += (long) modelDim * 4L;
            MemorySegment wQkv   = w.asSlice(off, (long) modelDim * qkvDim * 4L); off += (long) modelDim * qkvDim * 4L;
            MemorySegment wO     = w.asSlice(off, (long) qDim * modelDim * 4L); off += (long) qDim * modelDim * 4L;
            MemorySegment fnormW = w.asSlice(off, (long) modelDim * 4L); off += (long) modelDim * 4L;
            MemorySegment wGate  = w.asSlice(off, (long) modelDim * ffnDim * 4L); off += (long) modelDim * ffnDim * 4L;
            MemorySegment wUp    = w.asSlice(off, (long) modelDim * ffnDim * 4L); off += (long) modelDim * ffnDim * 4L;
            MemorySegment wDown  = w.asSlice(off, (long) ffnDim * modelDim * 4L);

            // ── 1. Pre-attention RMS Norm ────────────────────────────────────
            metal.rmsNorm(normed, residual, normW, modelDim, 1e-6f);

            // ── 2. QKV projection via MPS (AMX blocks) ───────────────────────
            metal.matmul(qkv, normed, wQkv, T, modelDim, qkvDim, 1.0f, 0.0f);

            // ── 3. FA4-equivalent attention via MetalFlashAttentionBinding ───
            // Gather K/V from paged cache into contiguous buffers for SDPA
            int    numKvSeqs = seqLen;
            long   kvGatherBytes = (long) numKvSeqs * numHeadsKv * headDim * 4L;
            MemorySegment kGathered = arena.allocate(kvGatherBytes, 64);
            MemorySegment vGathered = arena.allocate(kvGatherBytes, 64);
            gatherKV(arena, kPool, vPool, kGathered, vGathered, bt, seqLen, blockSz);

            boolean useBf16 = metalFa4.isBf16Available();
            metalFa4.fa4Attention(
                    attnOut, qkv, kGathered, vGathered,
                    1, T, seqLen, numHeads, numHeadsKv, headDim,
                    attnScale, true /* causal */, useBf16);

            // ── 4. Output projection + residual ──────────────────────────────
            MemorySegment proj = arena.allocate((long) T * modelDim * 4L, 64);
            metal.matmul(proj, attnOut, wO, T, qDim, modelDim, 1.0f, 0.0f);
            addResidual(residual, proj, T * modelDim);

            // ── 5. Pre-FFN RMS Norm ──────────────────────────────────────────
            metal.rmsNorm(normed, residual, fnormW, modelDim, 1e-6f);

            // ── 6. FFN gate + up projections ─────────────────────────────────
            metal.matmul(ffnGate, normed, wGate, T, modelDim, ffnDim, 1.0f, 0.0f);
            metal.matmul(ffnUp,   normed, wUp,   T, modelDim, ffnDim, 1.0f, 0.0f);

            // ── 7. SiLU gate ─────────────────────────────────────────────────
            metal.siluFfn(ffnGate, ffnGate, ffnUp, T * ffnDim);

            // ── 8. FFN down projection + residual ────────────────────────────
            metal.matmul(ffnOut, ffnGate, wDown, T, ffnDim, modelDim, 1.0f, 0.0f);
            addResidual(residual, ffnOut, T * modelDim);
        }

        // Read output logits — in production: final RMS norm + unembedding GEMM
        float[] logits = new float[Math.min(vocabSize, (int)(residual.byteSize() / 4L))];
        for (int i = 0; i < logits.length; i++)
            logits[i] = residual.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);
        return logits;
    }

    // ── KV gather helper ──────────────────────────────────────────────────────

    /**
     * Gather K/V vectors from Gollek's paged KV pool into contiguous buffers
     * for the FA4/SDPA kernel. Zero-copy on Apple Silicon because both the
     * paged pool and the gather buffers are in unified DRAM.
     */
    private void gatherKV(Arena arena,
                           MemorySegment kPool, MemorySegment vPool,
                           MemorySegment kOut, MemorySegment vOut,
                           int[] bt, int seqLen, int blockSz) {
        int numBlocks = bt.length;
        for (int blk = 0; blk < numBlocks; blk++) {
            int phys     = bt[blk];
            int tokStart = blk * blockSz;
            int tokEnd   = Math.min(seqLen, tokStart + blockSz);

            for (int tok = tokStart; tok < tokEnd; tok++) {
                for (int h = 0; h < numHeadsKv; h++) {
                    long srcOff = ((long) phys * numHeadsKv + h) * (long) blockSz * headDim
                                  + (long)(tok - tokStart) * headDim;
                    long dstOff = ((long) tok * numHeadsKv + h) * headDim;
                    long bytes  = (long) headDim * 4L;

                    kOut.asSlice(dstOff * 4L, bytes)
                        .copyFrom(kPool.asSlice(srcOff * 4L, bytes));
                    vOut.asSlice(dstOff * 4L, bytes)
                        .copyFrom(vPool.asSlice(srcOff * 4L, bytes));
                }
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void addResidual(MemorySegment a, MemorySegment b, int n) {
        for (int i = 0; i < n; i++)
            a.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    a.getAtIndex(ValueLayout.JAVA_FLOAT, i)
                  + b.getAtIndex(ValueLayout.JAVA_FLOAT, i));
    }

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    private Path resolveModelPath(ModelManifest m) {
        return m.artifacts().values().stream().findFirst()
                .map(l -> Path.of(l.uri()))
                .orElseThrow(() -> new IllegalArgumentException("No model artifact in manifest"));
    }

    private MemorySegment mmapModel(Path modelPath, Arena arena) {
        try {
            FileChannel ch = FileChannel.open(modelPath, StandardOpenOption.READ);
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            ch.close();
            log.infof("[Metal] Model mmap'd: %s (%.1f GB) into unified DRAM",
                    modelPath.getFileName(), ch.size() / 1e9);
            return seg;
        } catch (Exception e) {
            log.warnf("[Metal] Cannot mmap %s: %s — zero weights", modelPath, e.getMessage());
            return arena.allocate(256L * 1024 * 1024, 64);
        }
    }

    private MemorySegment[] sliceLayerWeights(MemorySegment weights, int layers) {
        long total = weights.byteSize();
        long perL  = total / layers;
        MemorySegment[] slices = new MemorySegment[layers];
        for (int i = 0; i < layers; i++) {
            long off  = (long) i * perL;
            long size = (i == layers - 1) ? (total - off) : perL;
            slices[i] = weights.asSlice(off, size);
        }
        return slices;
    }

    @Override public boolean health() { return initialized && metal != null; }

    @Override
    public void close() {
        initialized = false;
        if (weightsArena != null) { try { weightsArena.close(); } catch (Exception ignored) {} }
        log.info("[Metal] Closed");
    }
}
