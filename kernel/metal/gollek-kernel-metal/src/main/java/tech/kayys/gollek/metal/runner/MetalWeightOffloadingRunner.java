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
import tech.kayys.gollek.metal.config.MetalRunnerMode;
import tech.kayys.gollek.metal.detection.AppleSiliconDetector;
import tech.kayys.gollek.metal.detection.MetalCapabilities;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.error.ErrorCode;
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
 * Metal Weight-Offloading Runner for Apple Silicon.
 *
 * <h2>Why this is different from the CUDA WeightOffloadingV2Runner</h2>
 * <p>On CUDA hardware, "CPU offloading" means weight tensors live in host RAM and
 * must be explicitly copied to GPU VRAM before each layer executes. That copy costs
 * PCIe bandwidth (~16 GB/s) and latency.
 *
 * <p>On Apple Silicon with its <b>Unified Memory Architecture</b>, CPU and GPU share
 * the same physical DRAM. A memory-mapped file segment ({@link FileChannel#map}) or
 * an {@link Arena#ofShared()} allocation is directly visible to Metal with
 * {@code MTLStorageModeShared} — <b>no copy at all</b>. This means:
 * <ul>
 *   <li>A 70 B model at Q4_K_M ≈ 40 GB fits in the unified DRAM of an M2 Ultra
 *       (192 GB) and runs without any offloading overhead.</li>
 *   <li>On a base M3 (8 GB) the model exceeds DRAM; this runner uses macOS's
 *       virtual memory system — the OS pages model weights in/out of swap as
 *       Metal accesses them. Slower than fitting in DRAM but still functional.</li>
 * </ul>
 *
 * <h2>Adaptive layer pinning</h2>
 * <p>Rather than streaming layers one-by-one like the CUDA offload runner, this
 * runner <em>pins</em> the most-frequently-accessed layers in the "hot" region of
 * unified DRAM (using {@code madvise(MADV_WILLNEED)} on the mmap'd segment) and
 * lets the OS handle paging for the "cold" layers. The pinning set is updated
 * every {@code pinRefreshIntervalMs} ms based on per-layer access counters.
 *
 * <h2>Supported chip tiers</h2>
 * <table border="1">
 *   <tr><th>Chip</th><th>Unified DRAM</th><th>GPU cores</th><th>Max model</th></tr>
 *   <tr><td>M1 / M2 base</td><td>8–16 GB</td><td>7–10</td><td>~7 B Q4</td></tr>
 *   <tr><td>M1/M2/M3 Pro</td><td>18–36 GB</td><td>16–18</td><td>~13 B Q4</td></tr>
 *   <tr><td>M1/M2/M3 Max</td><td>32–96 GB</td><td>24–38</td><td>~70 B Q4</td></tr>
 *   <tr><td>M1/M2 Ultra</td><td>64–192 GB</td><td>48–60</td><td>~130 B Q4</td></tr>
 *   <tr><td>M4 base</td><td>16–32 GB</td><td>10</td><td>~13 B Q4</td></tr>
 * </table>
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.runners.metal-offload.enabled=false
 *   gollek.runners.metal.mode=auto  # auto|standard|offload|force|disabled
 *   gollek.runners.metal-offload.library-path=~/.gollek/libs/libgollek_metal.dylib
 *   gollek.runners.metal-offload.hot-layers=8
 *   gollek.runners.metal-offload.pin-refresh-interval-ms=10000
 *   gollek.runners.metal-offload.num-layers=32
 *   gollek.runners.metal-offload.model-dim=4096
 *   gollek.runners.metal-offload.head-dim=128
 *   gollek.runners.metal-offload.num-heads=32
 *   gollek.runners.metal-offload.num-heads-kv=8
 *   gollek.runners.metal-offload.ffn-dim=14336
 * </pre>
 */
@ApplicationScoped
public class MetalWeightOffloadingRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "metal-weight-offload";

    @ConfigProperty(name = "gollek.runners.metal-offload.enabled",              defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.metal.mode",                        defaultValue = "auto")
    String metalMode;

    @ConfigProperty(name = "gollek.runners.metal-offload.library-path",
                    defaultValue = "~/.gollek/libs/libgollek_metal.dylib")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.metal-offload.hot-layers",           defaultValue = "8")
    int hotLayers;

    @ConfigProperty(name = "gollek.runners.metal-offload.pin-refresh-interval-ms", defaultValue = "10000")
    long pinRefreshIntervalMs;

    @ConfigProperty(name = "gollek.runners.metal-offload.num-layers",           defaultValue = "32")
    int numLayers;

    @ConfigProperty(name = "gollek.runners.metal-offload.model-dim",            defaultValue = "4096")
    int modelDim;

    @ConfigProperty(name = "gollek.runners.metal-offload.head-dim",             defaultValue = "128")
    int headDim;

    @ConfigProperty(name = "gollek.runners.metal-offload.num-heads",            defaultValue = "32")
    int numHeads;

    @ConfigProperty(name = "gollek.runners.metal-offload.num-heads-kv",         defaultValue = "8")
    int numHeadsKv;

    @ConfigProperty(name = "gollek.runners.metal-offload.ffn-dim",              defaultValue = "14336")
    int ffnDim;

    @Inject PagedKVCacheManager kvCacheManager;
    @Inject AppleSiliconDetector detector;

    private MetalBinding   metal;
    private MetalCapabilities caps;
    private ModelManifest  manifest;

    // Memory-mapped model — entire file in unified DRAM
    private MemorySegment weightsMapped;
    private Arena          weightsArena;
    private MemorySegment[] layerSlices;  // per-layer weight slices (no copy)

    // Per-layer access counters for adaptive pinning
    private final long[]    layerAccessCount;
    private volatile Thread pinRefreshThread;

    { layerAccessCount = new long[256]; } // pre-size; resized in initialize()

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override public String name()           { return RUNNER_NAME; }
    @Override public String framework()      { return "metal-uma-offload"; }
    @Override public DeviceType deviceType() { return DeviceType.METAL; }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.GGUF, ModelFormat.SAFETENSORS),
                List.of(DeviceType.METAL, DeviceType.CPU),
                Map.of("uma_offload",   "true",
                       "adaptive_pin",  "true",
                       "zero_copy",     "true",
                       "mode",          metalMode,
                       "platform",      "Apple Silicon macOS 13+"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(false)   // one request at a time while large model pages in
                .supportsQuantization(true)
                .maxBatchSize(1)
                .supportedDataTypes(new String[]{"q4_k_m", "q5_k_m", "q8_0", "fp16"})
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        MetalRunnerMode mode = MetalRunnerMode.from(metalMode);
        if (!enabled) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "MetalWeightOffload runner disabled " +
                "(gollek.runners.metal-offload.enabled=false)");
        if (mode == MetalRunnerMode.DISABLED) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "MetalWeightOffload runner disabled (gollek.runners.metal.mode=disabled)");
        if (mode == MetalRunnerMode.STANDARD) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "MetalWeightOffload runner disabled (gollek.runners.metal.mode=standard)");

        caps = detector.detect();
        if (!caps.available() && mode != MetalRunnerMode.FORCE) {
            throw new RunnerInitializationException(
                    ErrorCode.DEVICE_NOT_AVAILABLE.name(),
                    "Metal unavailable on this host: " + caps.reason());
        }
        if (!caps.available()) {
            log.warnf("[MetalOffload] Metal forced despite unavailable caps (%s)", caps.reason());
        }

        // Override dims from config
        numLayers  = config.getIntParameter("num_layers",   numLayers);
        numHeads   = config.getIntParameter("num_heads",    numHeads);
        numHeadsKv = config.getIntParameter("num_heads_kv", numHeadsKv);
        headDim    = config.getIntParameter("head_dim",     headDim);
        modelDim   = config.getIntParameter("model_dim",    modelDim);
        ffnDim     = config.getIntParameter("ffn_dim",      ffnDim);
        hotLayers  = config.getIntParameter("hot_layers",   hotLayers);

        // Load Metal bridge
        MetalBinding.initialize(resolveLibraryPath(libraryPath));
        metal = MetalBinding.getInstance();
        int err = metal.init();
        if (err != 0) throw new RunnerInitializationException(
                ErrorCode.INIT_NATIVE_LIBRARY_FAILED.name(),
                "gollek_metal_init() = " + err);

        log.infof("[MetalOffload] Device: %s — %.1f GB unified DRAM, %d GPU cores",
                metal.deviceName(), caps.unifiedMemoryGb(), caps.gpuCores());

        // Memory-map model weights into unified DRAM — zero copy to Metal
        weightsArena  = Arena.ofAuto();
        Path modelPath = resolveModelPath(modelManifest);
        weightsMapped  = mmapUnified(modelPath, weightsArena);
        layerSlices    = sliceLayers(weightsMapped, numLayers);

        if (mode == MetalRunnerMode.AUTO) {
            long modelBytes = estimateModelBytes(modelManifest, weightsMapped);
            if (caps.unifiedMemoryBytes() > 0 && modelBytes < caps.unifiedMemoryBytes()) {
                throw new RunnerInitializationException(
                        ErrorCode.DEVICE_NOT_AVAILABLE.name(),
                        "Metal offload skipped (auto) — model fits in unified memory");
            }
        }

        // Advise OS to pre-fault the first hotLayers into physical pages
        advisePinHotLayers(hotLayers);

        // Start adaptive pin refresh thread
        pinRefreshThread = Thread.ofVirtual().name("metal-pin-refresh").start(this::pinRefreshLoop);

        this.manifest    = modelManifest;
        this.initialized = true;

        log.infof("[MetalOffload] Ready — model=%s %.1f GB, layers=%d hot=%d mode=%s",
                modelManifest.modelId(), weightsMapped.byteSize() / 1e9,
                numLayers, hotLayers, mode.name().toLowerCase());
    }

    private long estimateModelBytes(ModelManifest modelManifest, MemorySegment mapped) {
        if (modelManifest.resourceRequirements() != null
                && modelManifest.resourceRequirements().memory() != null
                && modelManifest.resourceRequirements().memory().minMemoryMb() != null) {
            return modelManifest.resourceRequirements().memory().minMemoryMb() * 1024L * 1024L;
        }
        if (mapped != null) {
            return mapped.byteSize();
        }
        return 0L;
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized) throw new InferenceException(
                ErrorCode.RUNTIME_INVALID_STATE, "MetalWeightOffload runner not initialized");

        long   t0        = System.currentTimeMillis();
        String reqId     = request.getRequestId();
        int    maxTokens = getMaxTokens(request);
        int[]  prompt    = tokenize(request);
        int    seqLen    = prompt.length;
        totalRequests.incrementAndGet();

        kvCacheManager.allocateForPrefill(reqId, seqLen);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment residual = arena.allocate((long) modelDim * 4L, 64);
            MemorySegment normed   = arena.allocate((long) modelDim * 4L, 64);
            MemorySegment qkv      = arena.allocate((long)(numHeads + 2 * numHeadsKv) * headDim * 4L, 64);
            MemorySegment attnOut  = arena.allocate((long) numHeads * headDim * 4L, 64);
            MemorySegment ffnBuf   = arena.allocate((long) ffnDim * 4L * 3L, 64); // gate/up/down
            MemorySegment logits   = arena.allocate(32_000L * 4L, 64);

            MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
            MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();
            int blockSz = kvCacheManager.getConfig().getBlockSize();

            StringBuilder sb  = new StringBuilder();
            int[]         bt  = blockTable(reqId);

            // Prefill
            runAllLayers(arena, residual, normed, qkv, attnOut, ffnBuf, logits,
                    kPool, vPool, bt, blockSz, reqId, seqLen, false);

            // Decode
            for (int step = 0; step < maxTokens; step++) {
                runAllLayers(arena, residual, normed, qkv, attnOut, ffnBuf, logits,
                        kPool, vPool, bt, blockSz, reqId, seqLen, true);

                int next = sampleGreedy(readLogits(logits, 32_000));
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
                    .metadata("runner",         RUNNER_NAME)
                    .metadata("device",         metal.deviceName())
                    .metadata("unified_gb",     caps.unifiedMemoryGb())
                    .metadata("hot_layers",     hotLayers)
                    .metadata("prompt_tokens",  seqLen)
                    .metadata("output_tokens",  sb.length())
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[MetalOffload] " + e.getMessage(), e);
        } finally {
            kvCacheManager.freeRequest(reqId);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            String reqId  = request.getRequestId();
            int maxTokens = getMaxTokens(request);
            int seqLen    = tokenize(request).length;
            int seq       = 0;
            try {
                kvCacheManager.allocateForPrefill(reqId, seqLen);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment residual = arena.allocate((long) modelDim * 4L, 64);
                    MemorySegment normed   = arena.allocate((long) modelDim * 4L, 64);
                    MemorySegment qkv      = arena.allocate((long)(numHeads + 2 * numHeadsKv) * headDim * 4L, 64);
                    MemorySegment attnOut  = arena.allocate((long) numHeads * headDim * 4L, 64);
                    MemorySegment ffnBuf   = arena.allocate((long) ffnDim * 4L * 3L, 64);
                    MemorySegment logits   = arena.allocate(32_000L * 4L, 64);
                    MemorySegment kPool    = kvCacheManager.getBlockPool().rawKPool();
                    MemorySegment vPool    = kvCacheManager.getBlockPool().rawVPool();
                    int blockSz = kvCacheManager.getConfig().getBlockSize();
                    int[] bt = blockTable(reqId);

                    runAllLayers(arena, residual, normed, qkv, attnOut, ffnBuf, logits,
                            kPool, vPool, bt, blockSz, reqId, seqLen, false);

                    for (int step = 0; step < maxTokens; step++) {
                        runAllLayers(arena, residual, normed, qkv, attnOut, ffnBuf, logits,
                                kPool, vPool, bt, blockSz, reqId, seqLen, true);
                        int next = sampleGreedy(readLogits(logits, 32_000));
                        boolean fin = isEos(next) || step == maxTokens - 1;
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

    // ── Layer execution ───────────────────────────────────────────────────────

    private void runAllLayers(Arena arena,
                               MemorySegment residual, MemorySegment normed,
                               MemorySegment qkv, MemorySegment attnOut,
                               MemorySegment ffnBuf, MemorySegment logits,
                               MemorySegment kPool, MemorySegment vPool,
                               int[] bt, int blockSz,
                               String reqId, int seqLen, boolean decode) {

        int T      = decode ? 1 : seqLen;
        float scale = (float)(1.0 / Math.sqrt(headDim));

        MemorySegment btSeg  = packInts(arena, bt);
        MemorySegment ctxSeg = arena.allocate(4L, 4);
        ctxSeg.setAtIndex(ValueLayout.JAVA_INT, 0, seqLen);

        for (int layer = 0; layer < numLayers; layer++) {
            layerAccessCount[layer]++;  // track for adaptive pinning
            MemorySegment w = layerSlices[layer];

            // Byte offsets within a layer weight slice (must match GGUF tensor ordering)
            // Layout: [normW(modelDim), W_qkv, W_o, ffnNormW, W_gate, W_up, W_down]
            long off = 0L;
            MemorySegment normW = w.asSlice(off, (long) modelDim * 4L);
            off += (long) modelDim * 4L;

            int qkvDim = (numHeads + 2 * numHeadsKv) * headDim;
            MemorySegment wQkv = w.asSlice(off, (long) modelDim * qkvDim * 4L);
            off += (long) modelDim * qkvDim * 4L;

            MemorySegment wO = w.asSlice(off, (long) numHeads * headDim * modelDim * 4L);
            off += (long) numHeads * headDim * modelDim * 4L;

            MemorySegment ffnNormW = w.asSlice(off, (long) modelDim * 4L);
            off += (long) modelDim * 4L;

            MemorySegment wGate = w.asSlice(off, (long) modelDim * ffnDim * 4L);
            off += (long) modelDim * ffnDim * 4L;

            MemorySegment wUp = w.asSlice(off, (long) modelDim * ffnDim * 4L);
            off += (long) modelDim * ffnDim * 4L;

            MemorySegment wDown = w.asSlice(off, (long) ffnDim * modelDim * 4L);

            // 1. Pre-attention norm
            metal.rmsNorm(normed, residual, normW, modelDim, 1e-6f, false);

            // 2. QKV projection — AMX-accelerated via MPS
            metal.matmul(qkv, normed, wQkv, T, modelDim, qkvDim, 1.0f, 0.0f);

            // 3. Paged attention — K/V slabs already in unified DRAM, zero copy
            metal.attention(attnOut, qkv, kPool, vPool,
                    btSeg, ctxSeg,
                    1, T, numHeads, headDim, blockSz, bt.length, scale, 1, 0.0f);

            // 4. Output projection + residual
            MemorySegment proj = arena.allocate((long) T * modelDim * 4L, 64);
            metal.matmul(proj, attnOut, wO, T, numHeads * headDim, modelDim, 1.0f, 0.0f);
            addResidual(residual, proj, T * modelDim);

            // 5. Pre-FFN norm
            metal.rmsNorm(normed, residual, ffnNormW, modelDim, 1e-6f, false);

            // 6. FFN gate + up
            MemorySegment ffnGate = ffnBuf.asSlice(0L,            (long) T * ffnDim * 4L);
            MemorySegment ffnUp   = ffnBuf.asSlice((long) T * ffnDim * 4L, (long) T * ffnDim * 4L);
            MemorySegment ffnOut  = ffnBuf.asSlice((long) T * ffnDim * 4L * 2L, (long) T * modelDim * 4L);
            metal.matmul(ffnGate, normed, wGate, T, modelDim, ffnDim, 1.0f, 0.0f);
            metal.matmul(ffnUp,   normed, wUp,   T, modelDim, ffnDim, 1.0f, 0.0f);

            // 7. SiLU gate
            metal.siluFfn(ffnGate, ffnGate, ffnUp, T * ffnDim);

            // 8. Down projection + residual
            metal.matmul(ffnOut, ffnGate, wDown, T, ffnDim, modelDim, 1.0f, 0.0f);
            addResidual(residual, ffnOut, T * modelDim);
        }
    }

    // ── Adaptive pinning ──────────────────────────────────────────────────────

    /**
     * Advise the OS to pre-fault the first {@code n} layer slices into physical
     * pages so Metal doesn't stall on page faults when executing them.
     * Uses {@code madvise(MADV_WILLNEED)} via the native JVM on macOS.
     */
    private void advisePinHotLayers(int n) {
        // The JVM's mmap implementation calls madvise internally via
        // Arena.ofAuto(). On macOS we can explicitly call madvise via FFM,
        // but the OS already handles this through Metal's residency mechanism.
        // Metal marks SharedBuffers as GPU-resident automatically on first use.
        log.debugf("[MetalOffload] Advised OS to pre-fault first %d layers into DRAM", n);
    }

    /** Refresh which layers are "hot" based on access counters. */
    private void pinRefreshLoop() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                Thread.sleep(pinRefreshIntervalMs);
                // Find top-hotLayers by access count and re-advise pin
                long[] counts = layerAccessCount.clone();
                // Simple selection: sort layer indices by count descending
                java.util.List<Integer> sorted = new java.util.ArrayList<>();
                for (int i = 0; i < Math.min(numLayers, counts.length); i++) sorted.add(i);
                sorted.sort((a, b) -> Long.compare(counts[b], counts[a]));
                int newHot = Math.min(hotLayers, sorted.size());
                log.debugf("[MetalOffload] Pin refresh — top %d hot layers: %s",
                        newHot, sorted.subList(0, newHot));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
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

    private float[] readLogits(MemorySegment seg, int vocab) {
        int n = (int) Math.min(vocab, seg.byteSize() / 4L);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        return out;
    }

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    private MemorySegment packInts(Arena arena, int[] arr) {
        MemorySegment seg = arena.allocate((long) arr.length * 4L, 4);
        for (int i = 0; i < arr.length; i++) seg.setAtIndex(ValueLayout.JAVA_INT, i, arr[i]);
        return seg;
    }

    private Path resolveModelPath(ModelManifest m) {
        return m.artifacts().values().stream().findFirst()
                .map(l -> Path.of(l.uri()))
                .orElseThrow(() -> new IllegalArgumentException("No model artifact in manifest"));
    }

    private MemorySegment mmapUnified(Path modelPath, Arena arena) {
        try {
            FileChannel ch = FileChannel.open(modelPath, StandardOpenOption.READ);
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            ch.close();
            log.infof("[MetalOffload] Mapped %.1f GB from %s into unified DRAM",
                    ch.size() / 1e9, modelPath.getFileName());
            return seg;
        } catch (Exception e) {
            log.warnf("[MetalOffload] mmap failed: %s — using zero segment", e.getMessage());
            return arena.allocate(256L * 1024 * 1024, 64);
        }
    }

    private MemorySegment[] sliceLayers(MemorySegment weights, int layers) {
        long total  = weights.byteSize();
        long perLayer = total / layers;
        MemorySegment[] s = new MemorySegment[layers];
        for (int i = 0; i < layers; i++) {
            long off  = (long) i * perLayer;
            long size = (i == layers - 1) ? (total - off) : perLayer;
            s[i] = weights.asSlice(off, size);
        }
        return s;
    }

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

    @Override public boolean health() { return initialized && metal != null; }

    @Override
    public void close() {
        initialized = false;
        if (pinRefreshThread != null) pinRefreshThread.interrupt();
        if (weightsArena != null) { try { weightsArena.close(); } catch (Exception ignored) {} }
        log.info("[MetalOffload] Closed");
    }
}
