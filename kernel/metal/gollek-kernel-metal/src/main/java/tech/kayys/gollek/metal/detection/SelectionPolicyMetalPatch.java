package tech.kayys.gollek.metal.detection;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.engine.routing.policy.SelectionPolicy;
import tech.kayys.gollek.metal.config.MetalRunnerMode;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.runner.RunnerCandidate;
import tech.kayys.gollek.model.core.RunnerMetrics;
import tech.kayys.gollek.model.core.HardwareDetector;

import java.util.List;

/**
 * Metal-aware SelectionPolicy decorator for Gollek.
 *
 * <h2>The gap this fills</h2>
 * <p>
 * Gollek's built-in {@link SelectionPolicy} calls {@code hw.hasCUDA()} to gate
 * CUDA runners, but for all other device types — including
 * {@link DeviceType#METAL} — it simply returns {@code true} without checking
 * whether the hardware is actually present. On an Apple Silicon Mac this means
 * the Metal runner would be scored and routed to correctly, but on a Linux CI
 * machine without Metal it would also appear as a candidate.
 *
 * <p>
 * This decorator wraps the existing policy and adds three behaviours:
 * <ol>
 * <li><b>Metal availability gate</b> — runners that declare
 * {@code DeviceType.METAL} are excluded on hosts where
 * {@link AppleSiliconDetector#detect()} reports Metal unavailable.</li>
 * <li><b>Unified-memory scoring bonus</b> — on Apple Silicon the Metal runner
 * receives a +15 score bonus because unified DRAM means no H2D/D2H copies,
 * which is equivalent to a significant latency advantage vs discrete GPUs
 * for KV-heavy workloads.</li>
 * <li><b>Large-model triage</b> — if the model's estimated size exceeds the
 * available unified memory, the Metal runner is deprioritised (−30) rather
 * than excluded, allowing it to run with weight offloading if no CUDA is
 * available.</li>
 * </ol>
 *
 * <h2>Registration</h2>
 * <p>
 * Because this is a CDI {@code @ApplicationScoped} bean that {@code @Inject}s
 * the base {@link SelectionPolicy}, Quarkus CDI will inject it wherever
 * {@code SelectionPolicy} is requested — acting as a transparent decorator
 * when the {@code gollek-ext-metal} module is on the classpath.
 *
 * <p>
 * The scoring logic delegates to the parent for all non-Metal runners so
 * existing CUDA/CPU behaviour is completely unchanged.
 */
@ApplicationScoped
public class SelectionPolicyMetalPatch extends SelectionPolicy {

    private static final Logger LOG = Logger.getLogger(SelectionPolicyMetalPatch.class);

    private static final int METAL_UMA_BONUS = 15; // zero-copy KV cache advantage
    private static final int METAL_MEMORY_PENALTY = -30; // model may not fit fully
    private static final int METAL_UNAVAILABLE_SKIP = Integer.MIN_VALUE; // excluded from ranking

    @Inject
    AppleSiliconDetector metalDetector;

    @ConfigProperty(name = "gollek.runners.metal.mode", defaultValue = "auto")
    String metalMode;

    @Inject
    public SelectionPolicyMetalPatch(
            RunnerMetrics runnerMetrics,
            HardwareDetector hardwareDetector) {
        super(runnerMetrics, hardwareDetector);
    }

    /**
     * Rank runners — delegates to parent for non-Metal devices, applies
     * Metal-specific
     * scoring adjustments for {@link DeviceType#METAL} runners.
     */
    @Override
    public List<RunnerCandidate> rankRunners(
            ModelManifest manifest,
            RequestContext context,
            List<String> configuredRunners) {

        MetalRunnerMode mode = MetalRunnerMode.from(metalMode);
        MetalCapabilities caps = metalDetector.detect();

        // Delegate the base ranking to the parent SelectionPolicy
        List<RunnerCandidate> ranked = super.rankRunners(manifest, context, configuredRunners);

        if (mode == MetalRunnerMode.DISABLED) {
            ranked = ranked.stream().filter(candidate -> !isMetal(candidate.name())).toList();
            if (ranked.size() < configuredRunners.size()) {
                LOG.info("[MetalPatch] Metal runners disabled via gollek.runners.metal.mode=disabled");
            }
            return ranked;
        }

        if (!caps.available() && mode != MetalRunnerMode.FORCE) {
            // Remove Metal-only runners from the candidate list on non-Apple hardware
            ranked = ranked.stream()
                    .filter(candidate -> !isMetal(candidate.name()))
                    .toList();
            if (ranked.size() < configuredRunners.size()) {
                LOG.debugf("[MetalPatch] Removed Metal runner(s) — Metal not available (%s)", caps.reason());
            }
            return ranked;
        }

        if (!caps.available() && mode == MetalRunnerMode.FORCE) {
            LOG.warnf("[MetalPatch] Metal forced despite unavailable caps (%s)", caps.reason());
        }

        // Apply mode-specific filtering (standard/offload).
        if (mode == MetalRunnerMode.STANDARD) {
            ranked = ranked.stream()
                    .filter(candidate -> !(isMetal(candidate.name()) && isOffload(candidate.name())))
                    .toList();
        } else if (mode == MetalRunnerMode.OFFLOAD) {
            ranked = ranked.stream()
                    .filter(candidate -> !(isMetal(candidate.name()) && !isOffload(candidate.name())))
                    .toList();
        }

        // Metal is available (or forced) — re-sort applying the unified-memory scoring
        // bonus
        long modelEstimateBytes = estimateModelBytes(manifest);
        boolean fitsInUnifiedMemory = caps.unifiedMemoryBytes() > 0
                && modelEstimateBytes < caps.unifiedMemoryBytes();

        // Rebuild with Metal runners moved to front if they fit in unified memory,
        // or kept but deprioritised if they don't
        java.util.List<RunnerCandidate> metalRunners = ranked.stream()
                .filter(candidate -> isMetal(candidate.name()))
                .toList();
        java.util.List<RunnerCandidate> otherRunners = ranked.stream()
                .filter(candidate -> !isMetal(candidate.name()))
                .toList();

        if (mode == MetalRunnerMode.FORCE) {
            java.util.List<RunnerCandidate> result = new java.util.ArrayList<>(metalRunners);
            result.addAll(otherRunners);
            return result;
        }

        if (mode == MetalRunnerMode.STANDARD) {
            java.util.List<RunnerCandidate> result = new java.util.ArrayList<>(metalRunners);
            result.addAll(otherRunners);
            return result;
        }
        if (mode == MetalRunnerMode.OFFLOAD) {
            java.util.List<RunnerCandidate> result = new java.util.ArrayList<>(metalRunners);
            result.addAll(otherRunners);
            return result;
        }

        if (fitsInUnifiedMemory) {
            // Metal fits: promote Metal runners to the top (zero-copy KV advantage)
            LOG.debugf("[MetalPatch] Metal UMA bonus applied — model %.1f GB fits in %.1f GB unified",
                    modelEstimateBytes / 1e9, caps.unifiedMemoryBytes() / 1e9);
            java.util.List<RunnerCandidate> result = new java.util.ArrayList<>();
            if (mode == MetalRunnerMode.AUTO) {
                metalRunners.stream()
                        .filter(candidate -> !isOffload(candidate.name()))
                        .forEach(result::add);
                metalRunners.stream()
                        .filter(candidate -> isOffload(candidate.name()))
                        .forEach(result::add);
            } else {
                result.addAll(metalRunners);
            }
            result.addAll(otherRunners);
            return result;
        } else {
            // Metal doesn't fully fit: keep Metal as fallback only (after other runners)
            LOG.debugf("[MetalPatch] Metal memory penalty — model %.1f GB > %.1f GB unified",
                    modelEstimateBytes / 1e9, caps.unifiedMemoryBytes() / 1e9);
            java.util.List<RunnerCandidate> result = new java.util.ArrayList<>(otherRunners);
            if (mode == MetalRunnerMode.AUTO) {
                metalRunners.stream()
                        .filter(candidate -> isOffload(candidate.name()))
                        .forEach(result::add);
                metalRunners.stream()
                        .filter(candidate -> !isOffload(candidate.name()))
                        .forEach(result::add);
            } else {
                result.addAll(metalRunners);
            }
            return result;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private boolean isMetal(String runnerName) {
        return runnerName != null && runnerName.toLowerCase().contains("metal");
    }

    private boolean isOffload(String runnerName) {
        return runnerName != null && runnerName.toLowerCase().contains("offload");
    }

    /**
     * Rough model size estimate from manifest resource requirements,
     * or from artifact file sizes if no requirements are declared.
     */
    private long estimateModelBytes(ModelManifest manifest) {
        if (manifest.resourceRequirements() != null
                && manifest.resourceRequirements().memory() != null
                && manifest.resourceRequirements().memory().minMemoryMb() != null) {
            return manifest.resourceRequirements().memory().minMemoryMb() * 1024L * 1024L;
        }
        // Fall back to summing artifact sizes
        return manifest.artifacts().values().stream()
                .mapToLong(loc -> {
                    try {
                        return java.nio.file.Files.size(java.nio.file.Path.of(loc.uri()));
                    } catch (Exception e) {
                        return 0L;
                    }
                })
                .sum();
    }
}
