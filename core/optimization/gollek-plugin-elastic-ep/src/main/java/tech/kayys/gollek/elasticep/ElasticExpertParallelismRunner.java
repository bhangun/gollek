package tech.kayys.gollek.elasticep;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.elasticep.binding.ElasticEpBinding;
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
import java.util.concurrent.atomic.AtomicLong;

/**
 * Elastic Expert Parallelism (MoE) Runner.
 *
 * <p>
 * Runs Mixture-of-Experts inference in-process across multiple GPUs via
 * {@link ElasticEpBinding}. Attention layers use Gollek's existing
 * {@link PagedAttentionBinding}. Expert dispatch and elastic rebalancing
 * are delegated to {@code libgollek_ep.so}.
 *
 * <h2>Paper: arXiv:2510.02613 (October 2025)</h2>
 * <ul>
 * <li><b>HMM zero-copy remapping</b> — VA→PA page table updates without data
 * movement.</li>
 * <li><b>HCCL NVLink P2P transfers</b> — weight migration via NVLink 4.0 (~900
 * GB/s).</li>
 * <li><b>VM-based rebalancing</b> — min-cost assignment every
 * {@code rebalanceIntervalMs} ms.</li>
 * </ul>
 * Results: 9× lower scale-up latency, 2× throughput during rebalancing, zero
 * downtime.
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.runners.elasticep.enabled=false
 *   gollek.runners.elasticep.library-path=/opt/gollek/lib/libgollek_ep.so
 *   gollek.runners.elasticep.num-experts=256
 *   gollek.runners.elasticep.num-gpus=8
 *   gollek.runners.elasticep.top-k=2
 *   gollek.runners.elasticep.rebalance-interval-ms=10000
 * </pre>
 */
@ApplicationScoped
public class ElasticExpertParallelismRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "elastic-ep-moe";

    @ConfigProperty(name = "gollek.runners.elasticep.enabled", defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.elasticep.library-path", defaultValue = "/opt/gollek/lib/libgollek_ep.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.elasticep.num-experts", defaultValue = "256")
    int numExperts;

    @ConfigProperty(name = "gollek.runners.elasticep.num-gpus", defaultValue = "8")
    int numGpus;

    @ConfigProperty(name = "gollek.runners.elasticep.top-k", defaultValue = "2")
    int topK;

    @ConfigProperty(name = "gollek.runners.elasticep.rebalance-interval-ms", defaultValue = "10000")
    long rebalanceIntervalMs;

    @Inject
    PagedKVCacheManager kvCacheManager;

    private ElasticEpBinding epBinding;
    private PagedAttentionBinding paBinding;
    private ModelManifest manifest;

    // Per-expert GPU assignment and load counters
    private volatile int[] expertGpuAssignment;
    private AtomicLong[] loadCounters;
    private volatile Thread rebalancerThread;

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "elastic-ep-cuda";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.SAFETENSORS),
                List.of(DeviceType.CUDA),
                Map.of("paper", "arXiv:2510.02613",
                        "mechanism", "HMM zero-copy + HCCL NVLink P2P",
                        "num_experts", String.valueOf(numExperts),
                        "num_gpus", String.valueOf(numGpus)));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(false)
                .maxBatchSize(256)
                .supportedDataTypes(new String[] { "bf16", "fp16" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "ElasticEP runner disabled (gollek.runners.elasticep.enabled=false)");

        numExperts = config.getIntParameter("num_experts", numExperts);
        numGpus = config.getIntParameter("num_gpus", numGpus);
        topK = config.getIntParameter("top_k", topK);

        // Load EP binding — delegates to ElasticEpCpuFallback if .so absent
        ElasticEpBinding.initialize(Path.of(libraryPath));
        epBinding = ElasticEpBinding.getInstance();

        // Reuse existing PagedAttentionBinding for attention sub-layers
        PagedAttentionBinding.initializeFallback();
        paBinding = PagedAttentionBinding.getInstance();

        // Initial round-robin expert → GPU assignment
        expertGpuAssignment = new int[numExperts];
        for (int i = 0; i < numExperts; i++)
            expertGpuAssignment[i] = i % numGpus;

        loadCounters = new AtomicLong[numExperts];
        for (int i = 0; i < numExperts; i++)
            loadCounters[i] = new AtomicLong();

        rebalancerThread = Thread.ofVirtual().name("ep-rebalancer").start(this::rebalanceLoop);

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[ElasticEP] Initialized — experts=%d gpus=%d topK=%d native=%s",
                numExperts, numGpus, topK, epBinding.isNativeAvailable());
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "ElasticEP runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTokens = getMaxTokens(request);
        int[] prompt = tokenize(request);
        int seqLen = prompt.length;
        totalRequests.incrementAndGet();

        kvCacheManager.allocateForPrefill(reqId, seqLen);
        try (Arena arena = Arena.ofConfined()) {
            int[] bt = blockTable(reqId);
            StringBuilder sb = new StringBuilder();

            // Prefill
            runMoELayer(arena, reqId, seqLen, bt);

            // Decode loop
            for (int step = 0; step < maxTokens; step++) {
                float[] logits = runMoELayer(arena, reqId, seqLen, bt);
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
                    .metadata("num_experts", numExperts)
                    .metadata("top_k", topK)
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[ElasticEP] " + e.getMessage(), e);
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

    // ── MoE layer ─────────────────────────────────────────────────────────────

    private float[] runMoELayer(Arena arena, String reqId, int seqLen, int[] bt) {
        int numH = 32;
        int hDim = 128;
        int dim = numH * hDim;
        float scale = (float) (1.0 / Math.sqrt(hDim));
        int bsz = kvCacheManager.getConfig().getBlockSize();

        // ── 1. Paged self-attention via existing PagedAttentionBinding ─────────
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

        // ── 2. Route tokens to topK experts via ElasticEpBinding ──────────────
        int[] selectedExperts = routeTopK(arena, out, topK);

        // Update load counters for rebalancer
        for (int eid : selectedExperts)
            if (eid >= 0 && eid < numExperts)
                loadCounters[eid].incrementAndGet();

        // Pack expert IDs into a MemorySegment
        MemorySegment expertIdsSeg = arena.allocate((long) topK * 4L, 4);
        for (int i = 0; i < topK; i++)
            expertIdsSeg.setAtIndex(ValueLayout.JAVA_INT, i, selectedExperts[i]);

        // ── 3. Dispatch via ElasticEpBinding ──────────────────────────────────
        MemorySegment epOut = arena.allocate((long) dim * 4L, 64);
        int err = epBinding.epDispatch(epOut, out, expertIdsSeg,
                1, seqLen, topK, numExperts, dim);
        if (err != 0)
            throw new RuntimeException("ep_dispatch error " + err);

        // Read logits from expert output
        float[] logits = new float[hDim];
        for (int i = 0; i < hDim; i++)
            logits[i] = epOut.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);
        return logits;
    }

    /** Simple round-robin router — replace with learned gating in production. */
    private int[] routeTopK(Arena arena, MemorySegment hidden, int k) {
        int[] ids = new int[k];
        long tick = System.nanoTime();
        for (int i = 0; i < k; i++)
            ids[i] = (int) ((tick + i) % numExperts);
        return ids;
    }

    // ── Elastic rebalancer ────────────────────────────────────────────────────

    private void rebalanceLoop() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                Thread.sleep(rebalanceIntervalMs);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment newAssign = arena.allocate((long) numExperts * 4L, 4);
                    MemorySegment histogram = arena.allocate((long) numExperts * 4L, 4);

                    for (int i = 0; i < numExperts; i++)
                        histogram.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                                (float) loadCounters[i].getAndSet(0));

                    int err = epBinding.epRebalance(newAssign, histogram, numExperts, numGpus);
                    if (err == 0) {
                        int[] updated = new int[numExperts];
                        for (int i = 0; i < numExperts; i++)
                            updated[i] = newAssign.getAtIndex(ValueLayout.JAVA_INT, i);

                        int[] prev = expertGpuAssignment;
                        int[][] remaps = computeRemaps(prev, updated);
                        for (int[] remap : remaps) {
                            epBinding.epHmmRemap(remap[0], remap[1], remap[2]);
                        }
                        expertGpuAssignment = updated;
                        log.debugf("[ElasticEP] Expert assignment rebalanced");
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                log.warnf("[ElasticEP] Rebalance error: %s", e.getMessage());
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    private int[][] computeRemaps(int[] prev, int[] updated) {
        if (prev == null || updated == null)
            return new int[0][0];
        int len = Math.min(prev.length, updated.length);
        int[][] tmp = new int[len][3];
        int count = 0;
        for (int e = 0; e < len; e++) {
            if (prev[e] != updated[e]) {
                tmp[count][0] = e;
                tmp[count][1] = prev[e];
                tmp[count][2] = updated[e];
                count++;
            }
        }
        int[][] out = new int[count][3];
        System.arraycopy(tmp, 0, out, 0, count);
        return out;
    }

    // Visible for tests
    int[][] computeRemapsForTest(int[] prev, int[] updated) {
        return computeRemaps(prev, updated);
    }

    @Override
    public boolean health() {
        return initialized && epBinding != null;
    }

    @Override
    public void close() {
        initialized = false;
        if (rebalancerThread != null)
            rebalancerThread.interrupt();
        log.info("[ElasticEP] Closed");
    }
}
