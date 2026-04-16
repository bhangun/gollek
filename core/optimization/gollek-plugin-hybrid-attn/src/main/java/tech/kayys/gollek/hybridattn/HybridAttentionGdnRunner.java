package tech.kayys.gollek.hybridattn;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.hybridattn.binding.GdnBinding;
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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Hybrid Attention + Gated Delta Network (GDN) Runner.
 *
 * <p>
 * Runs in-process inside Gollek using {@link GdnBinding} for GDN recurrent
 * layers and Gollek's existing {@link PagedAttentionBinding} for softmax
 * attention layers, interleaved per the H1/H2 schedule.
 *
 * <h2>Paper: arXiv:2412.06464 (ICLR 2025)</h2>
 * <p>
 * Gated delta rule: {@code S_t = α_t ⊙ S_{t-1} + β_t(v_t − S_{t-1}k_t)k_t^T}
 *
 * <h2>Hybrid configs</h2>
 * <ul>
 * <li><b>H1</b>: {GDN, GDN, GDN, GDN, ATTN, GDN} per 6 layers</li>
 * <li><b>H2</b>: {GDN, GDN, ATTN, GDN, GDN, ATTN} per 6 layers</li>
 * </ul>
 * H2 achieves 99.6% on MQAR retrieval vs Mamba2's 65.0%.
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.runners.hybridgdn.enabled=false
 *   gollek.runners.hybridgdn.library-path=/opt/gollek/lib/libgollek_gdn_kernels.so
 *   gollek.runners.hybridgdn.hybrid-config=H2
 *   gollek.runners.hybridgdn.num-layers=32
 *   gollek.runners.hybridgdn.state-dim=64
 *   gollek.runners.hybridgdn.head-dim=128
 * </pre>
 */
@ApplicationScoped
public class HybridAttentionGdnRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "hybrid-gdn";

    @ConfigProperty(name = "gollek.runners.hybridgdn.enabled", defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.hybridgdn.library-path", defaultValue = "/opt/gollek/lib/libgollek_gdn_kernels.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.hybridgdn.hybrid-config", defaultValue = "H2")
    String hybridConfig;

    @ConfigProperty(name = "gollek.runners.hybridgdn.num-layers", defaultValue = "32")
    int numLayers;

    @ConfigProperty(name = "gollek.runners.hybridgdn.state-dim", defaultValue = "64")
    int stateDim;

    @ConfigProperty(name = "gollek.runners.hybridgdn.head-dim", defaultValue = "128")
    int headDim;

    @ConfigProperty(name = "gollek.runners.hybridgdn.num-heads", defaultValue = "32")
    int numHeads;

    @Inject
    PagedKVCacheManager kvCacheManager;

    private GdnBinding gdnBinding;
    private PagedAttentionBinding paBinding;
    private ModelManifest manifest;
    private boolean[] layerIsGdn; // true = GDN layer, false = attention layer

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "hybrid-gdn-cuda";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.SAFETENSORS),
                List.of(DeviceType.CUDA, DeviceType.CPU),
                Map.of("paper", "arXiv:2412.06464",
                        "venue", "ICLR 2025",
                        "hybrid_config", hybridConfig,
                        "mqar_accuracy", "99.6%"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(false)
                .maxBatchSize(32)
                .supportedDataTypes(new String[] { "bf16", "fp32" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "HybridGDN runner disabled (gollek.runners.hybridgdn.enabled=false)");

        hybridConfig = config.getStringParameter("hybrid_config", hybridConfig);
        numLayers = config.getIntParameter("num_layers", numLayers);
        stateDim = config.getIntParameter("state_dim", stateDim);
        headDim = config.getIntParameter("head_dim", headDim);
        numHeads = config.getIntParameter("num_heads", numHeads);

        GdnBinding.initialize(Path.of(libraryPath));
        gdnBinding = GdnBinding.getInstance();

        PagedAttentionBinding.initializeFallback();
        paBinding = PagedAttentionBinding.getInstance();

        layerIsGdn = buildLayerSchedule(hybridConfig, numLayers);

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[HybridGDN] Initialized — config=%s layers=%d state_dim=%d native=%s",
                hybridConfig, numLayers, stateDim, gdnBinding.isNativeAvailable());
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode mixing GDN recurrent layers with paged softmax
     * attention.
     *
     * <p>
     * GDN layers update the persistent recurrent state {@code S} in-place;
     * attention layers read from Gollek's {@link PagedKVCacheManager} KV pool.
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "HybridGDN runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTok = getMaxTokens(request);
        int[] prompt = tokenize(request);
        int seqLen = prompt.length;
        int modelDim = numHeads * headDim;
        totalRequests.incrementAndGet();

        kvCacheManager.allocateForPrefill(reqId, seqLen);
        try (Arena arena = Arena.ofConfined()) {
            // GDN persistent state: [modelDim, stateDim] in fp32
            MemorySegment state = arena.allocate((long) modelDim * stateDim * 4L, 64);

            // Gate weight placeholders — in production loaded from model weights
            MemorySegment wAlpha = arena.allocate((long) modelDim * modelDim * 4L, 64);
            MemorySegment wBeta = arena.allocate((long) modelDim * modelDim * 4L, 64);

            StringBuilder sb = new StringBuilder();
            int[] bt = blockTable(reqId);

            // Prefill all layers
            runAllLayers(arena, state, wAlpha, wBeta, reqId, seqLen, bt, modelDim, false);

            // Decode loop
            for (int step = 0; step < maxTok; step++) {
                float[] logits = runAllLayers(
                        arena, state, wAlpha, wBeta, reqId, seqLen, bt, modelDim, true);
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
                    .metadata("hybrid_config", hybridConfig)
                    .metadata("prompt_tokens", prompt.length)
                    .metadata("output_tokens", sb.length())
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[HybridGDN] " + e.getMessage(), e);
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
            int modelDim = numHeads * headDim;
            int seq = 0;
            try {
                kvCacheManager.allocateForPrefill(reqId, seqLen);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment state = arena.allocate((long) modelDim * stateDim * 4L, 64);
                    MemorySegment wAlpha = arena.allocate((long) modelDim * modelDim * 4L, 64);
                    MemorySegment wBeta = arena.allocate((long) modelDim * modelDim * 4L, 64);
                    int[] bt = blockTable(reqId);

                    runAllLayers(arena, state, wAlpha, wBeta, reqId, seqLen, bt, modelDim, false);

                    for (int step = 0; step < maxTok; step++) {
                        float[] logits = runAllLayers(
                                arena, state, wAlpha, wBeta, reqId, seqLen, bt, modelDim, true);
                        int next = sampleGreedy(logits);
                        boolean fin = isEos(next) || step == maxTok - 1;
                        if (fin) {
                            emitter.emit(StreamingInferenceChunk.finalChunk(reqId, seq++, detokenize(next)));
                        } else {
                            emitter.emit(StreamingInferenceChunk.of(reqId, seq++, detokenize(next)));
                        }
                        if (fin)
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

    // ── Layer execution ───────────────────────────────────────────────────────

    /**
     * Run all {@code numLayers} transformer layers for one step.
     *
     * @param decodeOnly true = single-token decode; false = full-sequence prefill
     * @return output logits (headDim floats) from the last attention layer
     */
    private float[] runAllLayers(Arena arena,
            MemorySegment state,
            MemorySegment wAlpha, MemorySegment wBeta,
            String reqId, int seqLen, int[] bt,
            int modelDim, boolean decodeOnly) {
        int T = decodeOnly ? 1 : seqLen;
        float[] lastLogits = new float[headDim];

        for (int layer = 0; layer < numLayers; layer++) {
            if (layerIsGdn[layer]) {
                lastLogits = runGdnLayer(arena, state, wAlpha, wBeta, T, modelDim);
            } else {
                lastLogits = runAttentionLayer(arena, reqId, seqLen, bt);
            }
        }
        return lastLogits;
    }

    /** Run one GDN layer via {@link GdnBinding}. */
    private float[] runGdnLayer(Arena arena, MemorySegment state,
            MemorySegment wAlpha, MemorySegment wBeta,
            int T, int modelDim) {
        MemorySegment out = arena.allocate((long) T * modelDim * 4L, 64);
        MemorySegment input = arena.allocate((long) T * modelDim * 4L, 64);
        MemorySegment alpha = arena.allocate((long) T * modelDim * 4L, 64);
        MemorySegment beta = arena.allocate((long) T * modelDim * 4L, 64);

        // Compute gates via GdnBinding.gdnGateProject
        int err = gdnBinding.gdnGateProject(alpha, beta, input, wAlpha, wBeta, 1, T, modelDim);
        if (err != 0)
            throw new RuntimeException("gdn_gate_project error " + err);

        // Run GDN forward (updates state in-place)
        err = gdnBinding.gdnLayerForward(out, state, input, alpha, beta, 1, T, modelDim, stateDim);
        if (err != 0)
            throw new RuntimeException("gdn_layer_forward error " + err);

        float[] logits = new float[headDim];
        for (int i = 0; i < headDim; i++)
            logits[i] = out.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);
        return logits;
    }

    /**
     * Run one softmax attention layer via Gollek's existing
     * {@link PagedAttentionBinding}.
     */
    private float[] runAttentionLayer(Arena arena, String reqId, int seqLen, int[] bt) {
        int bsz = kvCacheManager.getConfig().getBlockSize();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        MemorySegment out = arena.allocate((long) numHeads * headDim * 4L, 64);
        MemorySegment query = arena.allocate((long) numHeads * headDim * 4L, 64);
        MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();
        MemorySegment btSeg = kvCacheManager.getBlockTableNative(reqId);
        MemorySegment ctxSeg = arena.allocate(4L, 4);
        ctxSeg.setAtIndex(ValueLayout.JAVA_INT, 0, seqLen);

        paBinding.pagedAttentionLaunch(
                out, query, kPool, vPool,
                btSeg, ctxSeg,
                1, numHeads, headDim, bsz, bt.length, scale);

        float[] logits = new float[headDim];
        for (int i = 0; i < headDim; i++)
            logits[i] = out.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);
        return logits;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Build the per-layer GDN/attention schedule.
     * H1: {T,T,T,T,F,T} per 6 layers; H2: {T,T,F,T,T,F} per 6 layers
     */
    static boolean[] buildLayerSchedule(String config, int layers) {
        boolean[] h1 = { true, true, true, true, false, true };
        boolean[] h2 = { true, true, false, true, true, false };
        boolean[] pat = "H1".equalsIgnoreCase(config) ? h1 : h2;
        boolean[] sched = new boolean[layers];
        for (int i = 0; i < layers; i++)
            sched[i] = pat[i % pat.length];
        return sched;
    }

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public boolean health() {
        return initialized && gdnBinding != null;
    }

    @Override
    public void close() {
        initialized = false;
        log.info("[HybridGDN] Closed");
    }
}
