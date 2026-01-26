package tech.kayys.golek.model.core;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.api.context.RequestContext;
import tech.kayys.golek.api.model.DeviceType;
import tech.kayys.golek.api.model.ModelFormat;
import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.api.model.RunnerMetadata;
import jakarta.enterprise.inject.Instance;

/**
 * Selection policy implementation with scoring algorithm
 */
@ApplicationScoped
public class SelectionPolicy {

    private final RunnerMetrics runnerMetrics;
    private final HardwareDetector hardwareDetector;

    @Inject
    Instance<ModelRunnerProvider> runnerProviders;

    @Inject
    public SelectionPolicy(
            RunnerMetrics runnerMetrics,
            HardwareDetector hardwareDetector) {
        this.runnerMetrics = runnerMetrics;
        this.hardwareDetector = hardwareDetector;
    }

    /**
     * Rank available runners based on multiple criteria
     */
    public List<RunnerCandidate> rankRunners(
            ModelManifest manifest,
            RequestContext context,
            List<String> configuredRunners) {
        List<RunnerCandidate> candidates = new ArrayList<>();

        // Get current hardware availability
        HardwareCapabilities hw = hardwareDetector.detect();

        for (String runnerName : configuredRunners) {
            RunnerMetadata runnerMeta = getRunnerMetadata(runnerName);
            if (runnerMeta == null)
                continue;

            // Filter by format compatibility
            if (!hasCompatibleFormat(manifest, runnerMeta)) {
                continue;
            }

            // Filter by device availability
            if (!isDeviceAvailable(runnerMeta, hw, context)) {
                continue;
            }

            // Calculate score
            int score = calculateScore(
                    manifest,
                    runnerMeta,
                    context,
                    hw);

            candidates.add(new RunnerCandidate(
                    runnerName,
                    score,
                    runnerMeta));
        }

        // Sort by score descending
        candidates.sort(Comparator.comparing(
                RunnerCandidate::score).reversed());

        return candidates;
    }

    private RunnerMetadata getRunnerMetadata(String runnerName) {
        return runnerProviders.stream()
                .map(ModelRunnerProvider::metadata)
                .filter(m -> m.name().equals(runnerName))
                .findFirst()
                .orElse(null);
    }

    private boolean hasCompatibleFormat(ModelManifest manifest, RunnerMetadata runner) {
        return manifest.artifacts().keySet().stream()
                .anyMatch(format -> runner.supportedFormats().contains(format));
    }

    private boolean isDeviceAvailable(RunnerMetadata runner, HardwareCapabilities hw, RequestContext context) {
        // If preferred device set, ensure runner supports it or can fall back
        if (context.preferredDevice().isPresent()) {
            DeviceType preferred = context.preferredDevice().get();
            if (runner.supportedDevices().contains(preferred)) {
                if (preferred == DeviceType.CUDA)
                    return hw.hasCUDA();
                return true;
            }
        }

        // Generic check: ensure runner supports at least one device available on host
        return runner.supportedDevices().stream()
                .anyMatch(device -> {
                    if (device == DeviceType.CUDA)
                        return hw.hasCUDA();
                    return true; // CPU, etc.
                });
    }

    private boolean hasAvailableResources(ModelManifest manifest, RunnerMetadata runner, HardwareCapabilities hw) {
        if (manifest.resourceRequirements() == null || manifest.resourceRequirements().minMemory() == null) {
            return true;
        }

        long reqMemory = manifest.resourceRequirements().minMemory().asLongValue();
        return hw.getAvailableMemory() >= reqMemory;
    }

    /**
     * Multi-factor scoring algorithm
     */
    private int calculateScore(
            ModelManifest manifest,
            RunnerMetadata runner,
            RequestContext context,
            HardwareCapabilities hw) {
        int score = 0;

        // 1. Device preference match (highest weight)
        if (context.preferredDevice().isPresent()) {
            DeviceType preferred = context.preferredDevice().get();
            if (runner.supportedDevices().contains(preferred)) {
                score += 50;
            }
        }

        // 2. Format native support
        if (!manifest.artifacts().isEmpty()) {
            ModelFormat firstFormat = manifest.artifacts().keySet().iterator().next();
            if (runner.supportedFormats().contains(firstFormat)) {
                score += 30;
            }
        }

        // 3. Historical performance (P95 latency)
        Optional<Duration> p95 = runnerMetrics.getP95Latency(
                runner.name(),
                manifest.modelId());
        if (p95.isPresent() && p95.get().compareTo(context.timeout()) < 0) {
            score += 25;
        }

        // 4. Resource availability
        if (hasAvailableResources(manifest, runner, hw)) {
            score += 20;
        }

        // 5. Health status
        if (runnerMetrics.isHealthy(runner.name())) {
            score += 15;
        }

        // 6. Cost optimization (favor CPU over GPU if performance OK)
        if (context.costSensitive() && runner.supportedDevices().contains(DeviceType.CPU)) {
            score += 10;
        }

        // 7. Current load (avoid overloaded runners)
        double currentLoad = runnerMetrics.getCurrentLoad(runner.name());
        if (currentLoad < 0.7) {
            score += 15;
        } else if (currentLoad > 0.95) {
            score -= 50; // Heavy penalty for very high load
        } else if (currentLoad > 0.8) {
            score -= 20;
        }

        return score;
    }
}