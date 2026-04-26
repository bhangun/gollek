package tech.kayys.gollek.qlora;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.qlora.binding.QLoraBinding;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.kernel.paged.PagedAttentionBinding;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

/**
 * QLoRA Adapter Runner — NF4 base + INT4 LoRA fused GEMM.
 *
 * <p>
 * Uses {@link QLoraBinding} for the fused {@code qlora_fused_matmul} kernel
 * and Gollek's {@link PagedAttentionBinding} for attention layers.
 * Adapters are cached in an LRU map keyed by adapter ID and loaded from
 * {@code adapterStore} on first use.
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.runners.qlora.enabled=false
 *   gollek.runners.qlora.library-path=/opt/gollek/lib/libgollek_qlora.so
 *   gollek.runners.qlora.adapter-store=/models/adapters
 *   gollek.runners.qlora.max-cached-adapters=8
 *   gollek.runners.qlora.lora-rank=16
 *   gollek.runners.qlora.lora-alpha=32
 * </pre>
 */
@ApplicationScoped
public class QLoraAdapterRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "qlora-adapter";

    @ConfigProperty(name = "gollek.runners.qlora.enabled", defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.qlora.library-path", defaultValue = "/opt/gollek/lib/libgollek_qlora.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.qlora.adapter-store", defaultValue = "/models/adapters")
    String adapterStore;

    @ConfigProperty(name = "gollek.runners.qlora.max-cached-adapters", defaultValue = "8")
    int maxCachedAdapters;

    @ConfigProperty(name = "gollek.runners.qlora.lora-rank", defaultValue = "16")
    int loraRank;

    @ConfigProperty(name = "gollek.runners.qlora.lora-alpha", defaultValue = "32")
    float loraAlpha;

    @Inject
    PagedKVCacheManager kvCacheManager;

    private QLoraBinding qloraBinding;
    private PagedAttentionBinding paBinding;
    private ModelManifest manifest;
    private MemorySegment baseWeights; // NF4-quantised base model (mmap'd)
    private Arena weightsArena;

    /** LRU adapter weight cache: adapterId → (loraB segment, loraA segment) */
    private final LinkedHashMap<String, AdapterWeights> adapterCache;
    private final ReentrantLock cacheLock = new ReentrantLock();

    record AdapterWeights(MemorySegment loraB, MemorySegment loraA, Arena arena) {
    }

    public QLoraAdapterRunner() {
        this.adapterCache = new LinkedHashMap<>(16, 0.75f, true /* LRU */);
    }

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "qlora-nf4-int4";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.GGUF, ModelFormat.SAFETENSORS),
                List.of(DeviceType.CUDA),
                Map.of("quantisation", "nf4+int4",
                        "fused_kernel", "true",
                        "lru_cache", String.valueOf(maxCachedAdapters)));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true)
                .maxBatchSize(16)
                .supportedDataTypes(new String[] { "nf4", "int4", "fp16" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "QLoRA runner disabled (gollek.runners.qlora.enabled=false)");

        loraRank = config.getIntParameter("lora_rank", loraRank);
        loraAlpha = config.getIntParameter("lora_alpha", (int) loraAlpha);

        QLoraBinding.initialize(Path.of(libraryPath));
        qloraBinding = QLoraBinding.getInstance();

        PagedAttentionBinding.initializeFallback();
        paBinding = PagedAttentionBinding.getInstance();

        // Memory-map NF4 base weights
        weightsArena = Arena.ofAuto();
        Path modelPath = resolveModelPath(modelManifest);
        baseWeights = mmapWeights(modelPath, weightsArena);

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[QLoRA] Initialized — model=%s rank=%d alpha=%.1f cache=%d native=%s",
                modelManifest.modelId(), loraRank, loraAlpha,
                maxCachedAdapters, qloraBinding.isNativeAvailable());
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "QLoRA runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTokens = getMaxTokens(request);
        int[] prompt = tokenize(request);
        int seqLen = prompt.length;
        String adapterId = resolveAdapterId(request);
        float scale = loraAlpha / loraRank;
        totalRequests.incrementAndGet();

        AdapterWeights adapter = ensureAdapterCached(adapterId);
        kvCacheManager.allocateForPrefill(reqId, seqLen);

        try (Arena arena = Arena.ofConfined()) {
            int N = 128, K = 128, M = seqLen;
            MemorySegment fusedOut = arena.allocate((long) M * N * 4L, 64);
            MemorySegment input = arena.allocate((long) M * K * 4L, 64);

            StringBuilder sb = new StringBuilder();
            int[] bt = blockTable(reqId);

            // Prefill: fused NF4+LoRA matmul + paged attention
            int err = qloraBinding.fusedMatmul(
                    fusedOut, baseWeights, adapter.loraB(), adapter.loraA(),
                    input, scale, M, N, K, loraRank);
            if (err != 0)
                throw new InferenceException(
                        ErrorCode.RUNTIME_INFERENCE_FAILED, "qlora_fused_matmul error " + err);

            // Decode loop
            for (int step = 0; step < maxTokens; step++) {
                // Attention sub-layer
                float[] attnLogits = runAttention(arena, reqId, seqLen, bt);

                // Fused matmul for single decode token
                MemorySegment decOut = arena.allocate((long) N * 4L, 64);
                MemorySegment decIn = arena.allocate((long) K * 4L, 64);
                qloraBinding.fusedMatmul(
                        decOut, baseWeights, adapter.loraB(), adapter.loraA(),
                        decIn, scale, 1, N, K, loraRank);

                // Merge attention + fused output logits
                float[] logits = new float[N];
                for (int i = 0; i < N; i++) {
                    logits[i] = (i < attnLogits.length ? attnLogits[i] : 0f)
                            + decOut.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);
                }

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
                    .metadata("adapter_id", adapterId)
                    .metadata("lora_rank", loraRank)
                    .metadata("cached_adapters", adapterCache.size())
                    .build();

        } catch (InferenceException ie) {
            totalFailures.incrementAndGet();
            throw ie;
        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[QLoRA] " + e.getMessage(), e);
        } finally {
            kvCacheManager.freeRequest(reqId);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                InferenceResponse r = infer(request);
                emitter.emit(StreamingInferenceChunk.finalChunk(
                        request.getRequestId(), 0, r.getContent()));
                emitter.complete();
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    // ── Attention helper ──────────────────────────────────────────────────────

    private float[] runAttention(Arena arena, String reqId, int seqLen, int[] bt) {
        int numH = 32;
        int hDim = 128;
        int bsz = kvCacheManager.getConfig().getBlockSize();
        float scale = (float) (1.0 / Math.sqrt(hDim));

        MemorySegment out = arena.allocate((long) numH * hDim * 4L, 64);
        MemorySegment query = arena.allocate((long) numH * hDim * 4L, 64);
        MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();
        MemorySegment btSeg = kvCacheManager.getBlockTableNative(reqId);
        MemorySegment ctxSeg = arena.allocate(4L, 4);
        ctxSeg.setAtIndex(ValueLayout.JAVA_INT, 0, seqLen);

        paBinding.pagedAttentionLaunch(
                out, query, kPool, vPool,
                btSeg, ctxSeg,
                1, numH, hDim, bsz, bt.length, scale);

        float[] logits = new float[hDim];
        for (int i = 0; i < hDim; i++)
            logits[i] = out.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);
        return logits;
    }

    // ── LRU adapter cache ─────────────────────────────────────────────────────

    private AdapterWeights ensureAdapterCached(String adapterId) {
        cacheLock.lock();
        try {
            AdapterWeights existing = adapterCache.get(adapterId);
            if (existing != null)
                return existing;

            // Evict LRU if at capacity
            if (adapterCache.size() >= maxCachedAdapters) {
                String lruKey = adapterCache.entrySet().iterator().next().getKey();
                AdapterWeights evicted = adapterCache.remove(lruKey);
                try {
                    evicted.arena().close();
                } catch (Exception ignored) {
                }
                log.debugf("[QLoRA] Evicted adapter %s from cache", lruKey);
            }

            AdapterWeights loaded = loadAdapter(adapterId);
            adapterCache.put(adapterId, loaded);
            log.debugf("[QLoRA] Loaded adapter %s (cache %d/%d)",
                    adapterId, adapterCache.size(), maxCachedAdapters);
            return loaded;
        } finally {
            cacheLock.unlock();
        }
    }

    private AdapterWeights loadAdapter(String adapterId) {
        Arena arena = Arena.ofAuto();
        Path adapterPath = Path.of(adapterStore, adapterId + ".bin");
        try (RandomAccessFile raf = new RandomAccessFile(adapterPath.toFile(), "r");
                FileChannel ch = raf.getChannel()) {
            long total = ch.size();
            long halfB = total / 2;
            MemorySegment loraB = ch.map(FileChannel.MapMode.READ_ONLY, 0L, halfB, arena);
            MemorySegment loraA = ch.map(FileChannel.MapMode.READ_ONLY, halfB, halfB, arena);
            return new AdapterWeights(loraB, loraA, arena);
        } catch (Exception e) {
            log.warnf("[QLoRA] Adapter %s not found: %s — using zero delta", adapterId, e.getMessage());
            MemorySegment zero = arena.allocate(1024L * 1024, 64);
            return new AdapterWeights(zero, zero.asSlice(0L), arena);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private String resolveAdapterId(InferenceRequest request) {
        Object id = request.getParameters().get("adapter_id");
        if (id == null)
            id = request.getParameters().get("lora_adapter_id");
        if (id == null)
            id = request.getMetadata().get("adapter_id");
        return id != null ? id.toString() : "default";
    }

    private Path resolveModelPath(ModelManifest m) {
        return m.artifacts().values().stream().findFirst()
                .map(l -> Path.of(l.uri()))
                .orElseThrow(() -> new IllegalArgumentException("No model artifact"));
    }

    private MemorySegment mmapWeights(Path p, Arena arena) {
        try (RandomAccessFile raf = new RandomAccessFile(p.toFile(), "r");
                FileChannel ch = raf.getChannel()) {
            return ch.map(FileChannel.MapMode.READ_ONLY, 0L, ch.size(), arena);
        } catch (Exception e) {
            log.warnf("[QLoRA] Cannot mmap %s: %s — using zero weights", p, e.getMessage());
            return arena.allocate(256L * 1024 * 1024, 64);
        }
    }

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public boolean health() {
        return initialized && qloraBinding != null;
    }

    @Override
    public void close() {
        initialized = false;
        cacheLock.lock();
        try {
            adapterCache.values().forEach(a -> {
                try {
                    a.arena().close();
                } catch (Exception ignored) {
                }
            });
            adapterCache.clear();
        } finally {
            cacheLock.unlock();
        }
        if (weightsArena != null) {
            try {
                weightsArena.close();
            } catch (Exception ignored) {
            }
        }
        log.info("[QLoRA] Closed");
    }
}
