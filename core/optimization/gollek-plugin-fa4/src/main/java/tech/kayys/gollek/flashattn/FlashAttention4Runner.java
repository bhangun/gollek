package tech.kayys.gollek.flashattn;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.flashattn.binding.FlashAttention4Binding;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;
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
 * FlashAttention-4 ModelRunner for NVIDIA Blackwell (sm_100a).
 *
 * <p>
 * Runs transformer inference <b>in-process</b> inside Gollek using
 * {@link FlashAttention4Binding} — a dedicated FFM binding to
 * {@code libgollek_fa4_kernels.so}, compiled from
 * {@code src/main/cpp/fa4/gollek_fa4_kernels.cu}.
 *
 * <h2>Paper: arXiv:2603.05451 (March 2026)</h2>
 * <p>
 * Three innovations over FA3 targeting Blackwell's asymmetric hardware:
 * <ul>
 * <li><b>Async UMMA pipelines in TMEM</b> — {@code tcgen05.mma} accumulates
 * S=QKᵀ in TMEM (256 KB/SM); two 128-token Q tiles ping-pong so
 * matmul and softmax/memory overlap fully.</li>
 * <li><b>Software exp() on FMA</b> — polynomial approximation on FMA units
 * bypasses the MUFU bottleneck that limits FA3 on Blackwell.</li>
 * <li><b>2-CTA MMA backward</b> — dS stored in TMEM; DSMEM resolves the
 * dQ reduction without shared-memory traffic.</li>
 * </ul>
 * Result on B200 BF16: 1 613 TFLOPs/s (71 % peak), 1.3× cuDNN 9.13.
 *
 * <h2>Integration</h2>
 * <p>
 * Discovered by {@code ModelRunnerFactory} via CDI
 * {@code @Inject Instance<ModelRunner>}. KV blocks are managed by Gollek's
 * existing {@link PagedKVCacheManager}; the FA4 kernel reads K/V directly
 * from {@link tech.kayys.gollek.kvcache.PhysicalBlockPool} off-heap slabs.
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.runners.fa4.enabled=false
 *   gollek.runners.fa4.library-path=/opt/gollek/lib/libgollek_fa4_kernels.so
 *   gollek.runners.fa4.num-heads=32
 *   gollek.runners.fa4.num-heads-kv=8
 *   gollek.runners.fa4.head-dim=128
 *   gollek.runners.fa4.use-fp8=false
 *   gollek.runners.fa4.causal=true
 * </pre>
 *
 * <h3>Build native kernel</h3>
 * 
 * <pre>
 *   make -C src/main/cpp/fa4   # requires CUDA 12.8+, sm_100a
 * </pre>
 */
@ApplicationScoped
public class FlashAttention4Runner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "flashattn4-blackwell";

    @ConfigProperty(name = "gollek.runners.fa4.enabled", defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.fa4.library-path", defaultValue = "/opt/gollek/lib/libgollek_fa4_kernels.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.fa4.num-heads", defaultValue = "32")
    int numHeads;

    @ConfigProperty(name = "gollek.runners.fa4.num-heads-kv", defaultValue = "8")
    int numHeadsKv;

    @ConfigProperty(name = "gollek.runners.fa4.head-dim", defaultValue = "128")
    int headDim;

    @ConfigProperty(name = "gollek.runners.fa4.use-fp8", defaultValue = "false")
    boolean useFp8;

    @ConfigProperty(name = "gollek.runners.fa4.causal", defaultValue = "true")
    boolean causal;

    @Inject
    PagedKVCacheManager kvCacheManager;

    private FlashAttention4Binding fa4Binding;
    private ModelManifest manifest;
    private Arena weightsArena;

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "flashattn4-cuda";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "4.0.0",
                List.of(ModelFormat.GGUF, ModelFormat.SAFETENSORS),
                List.of(DeviceType.CUDA),
                Map.of("arch", "sm_100a",
                        "paper", "arXiv:2603.05451",
                        "peak_tflops", "1613",
                        "peak_pct", "71",
                        "kernel", "CuTe-DSL UMMA/TMEM"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true)
                .maxBatchSize(64)
                .supportedDataTypes(new String[] { "bf16", "fp16", "fp8" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "FA4 runner disabled (gollek.runners.fa4.enabled=false)");

        // Load FA4-specific binding (dedicated class, not FA3)
        FlashAttention4Binding.initialize(Path.of(libraryPath));
        this.fa4Binding = FlashAttention4Binding.getInstance();

        // Override dims from per-model config
        numHeads = config.getIntParameter("num_heads", numHeads);
        numHeadsKv = config.getIntParameter("num_heads_kv", numHeadsKv);
        headDim = config.getIntParameter("head_dim", headDim);

        // Memory-map model weights
        weightsArena = Arena.ofAuto();
        Path modelPath = resolveModelPath(modelManifest);
        mmapWeights(modelPath, weightsArena);

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[FA4] Initialized — model=%s heads=%d/%d dim=%d fp8=%s native=%s",
                modelManifest.modelId(), numHeads, numHeadsKv, headDim,
                useFp8, fa4Binding.isNativeAvailable());
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode with FA4 attention in Gollek's paged KV memory.
     *
     * <p>
     * Prefill: one FA4 kernel call covering all prompt tokens, writing K/V
     * vectors into the paged pool. Decode: per-step kernel calls with a single
     * query token, extending blocks via {@link PagedKVCacheManager#appendToken}.
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "FA4 runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTokens = getMaxTokens(request);
        int[] prompt = tokenize(request);
        int promptLen = prompt.length;
        totalRequests.incrementAndGet();

        kvCacheManager.allocateForPrefill(reqId, promptLen);
        try (Arena arena = Arena.ofConfined()) {
            int[] bt = blockTable(reqId);
            StringBuilder sb = new StringBuilder();
            int seqLen = promptLen;

            // Prefill — fills K/V cache for all prompt positions
            runKernel(arena, seqLen, bt, false);

            // Decode loop
            for (int step = 0; step < maxTokens; step++) {
                int next = runKernel(arena, seqLen, bt, true);
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
                    .metadata("prompt_tokens", promptLen)
                    .metadata("output_tokens", sb.length())
                    .metadata("kernel", "fa4-umma-tmem-sm100a")
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[FA4] " + e.getMessage(), e);
        } finally {
            kvCacheManager.freeRequest(reqId);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            String reqId = request.getRequestId();
            int maxTokens = getMaxTokens(request);
            int seqLen = tokenize(request).length;
            int seq = 0;
            try {
                kvCacheManager.allocateForPrefill(reqId, seqLen);
                try (Arena arena = Arena.ofConfined()) {
                    int[] bt = blockTable(reqId);
                    runKernel(arena, seqLen, bt, false); // prefill
                    for (int step = 0; step < maxTokens; step++) {
                        int next = runKernel(arena, seqLen, bt, true);
                        boolean f = isEos(next) || step == maxTokens - 1;
                        if (f) {
                            emitter.emit(StreamingInferenceChunk.finalChunk(reqId, seq++, detokenize(next)));
                        } else {
                            emitter.emit(StreamingInferenceChunk.of(reqId, seq++, detokenize(next)));
                        }
                        if (f)
                            break;
                        seqLen++;
                        if (kvCacheManager.appendToken(reqId))
                            bt = blockTable(reqId);
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

    // ── Kernel dispatch ───────────────────────────────────────────────────────

    /**
     * Call the FA4 CUDA kernel via
     * {@link FlashAttention4Binding#flashAttention4Launch}.
     *
     * <p>
     * K/V slabs come from
     * {@link tech.kayys.gollek.kvcache.PhysicalBlockPool#rawKPool()}
     * and {@code rawVPool()} — the same off-heap segments that all other Gollek
     * runners use.
     *
     * @param decodeOnly true = single-query decode step; false = full-sequence
     *                   prefill
     * @return greedy next-token id (decode only) or 0 (prefill)
     */
    private int runKernel(Arena arena, int seqLen, int[] bt, boolean decodeOnly) {
        int batchSize = 1;
        int T = decodeOnly ? 1 : seqLen;
        float scale = (float) (1.0 / Math.sqrt(headDim));
        long elemBytes = useFp8 ? 1L : 2L;

        MemorySegment query = arena.allocate(
                (long) batchSize * T * numHeads * headDim * elemBytes, 64);
        MemorySegment output = arena.allocate(
                (long) batchSize * T * numHeads * headDim * elemBytes, 64);

        // K/V pool slabs come directly from PhysicalBlockPool — zero copy
        MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();

        int err = fa4Binding.flashAttention4Launch(
                output, query, kPool, vPool,
                batchSize, seqLen, numHeads, numHeadsKv, headDim,
                scale, causal, useFp8);

        if (err != 0)
            throw new RuntimeException("FA4 kernel error code " + err);

        if (decodeOnly) {
            float[] logits = new float[headDim];
            for (int d = 0; d < headDim; d++)
                logits[d] = output.getAtIndex(ValueLayout.JAVA_FLOAT, (long) d);
            return sampleGreedy(logits);
        }
        return 0;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId)
                .stream().mapToInt(Integer::intValue).toArray();
    }

    private Path resolveModelPath(ModelManifest m) {
        return m.artifacts().values().stream().findFirst()
                .map(l -> Path.of(l.uri()))
                .orElseThrow(() -> new IllegalArgumentException("No model artifact in manifest"));
    }

    private MemorySegment mmapWeights(Path modelPath, Arena arena) {
        try {
            FileChannel ch = FileChannel.open(modelPath, StandardOpenOption.READ);
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            ch.close();
            log.infof("[FA4] Model mmap'd: %s (%.1f GB)", modelPath.getFileName(), ch.size() / 1e9);
            return seg;
        } catch (Exception e) {
            log.warnf("[FA4] Cannot mmap %s: %s — zero weights", modelPath, e.getMessage());
            return arena.allocate(256L * 1024 * 1024, 64);
        }
    }

    @Override
    public boolean health() {
        return initialized && fa4Binding != null;
    }

    @Override
    public void close() {
        initialized = false;
        if (weightsArena != null) {
            try {
                weightsArena.close();
            } catch (Exception ignored) {
            }
        }
        log.info("[FA4] Closed");
    }
}
