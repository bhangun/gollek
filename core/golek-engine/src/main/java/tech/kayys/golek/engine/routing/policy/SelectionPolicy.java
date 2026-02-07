package tech.kayys.golek.engine.routing.policy;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.eclipse.microprofile.config.inject.ConfigProperty;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.spi.context.RequestContext;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.model.DeviceType;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.RunnerMetadata;
import tech.kayys.golek.spi.provider.RoutingContext;
import tech.kayys.golek.spi.provider.ProviderCandidate;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.model.core.RunnerMetrics;
import tech.kayys.golek.engine.model.RunnerCandidate;
import tech.kayys.golek.engine.model.ModelRunnerProvider;
import tech.kayys.golek.model.core.HardwareDetector;
import tech.kayys.golek.model.core.HardwareCapabilities;
import jakarta.enterprise.inject.Instance;
import org.jboss.logging.Logger;

/**
 * Selection policy implementation with scoring algorithm
 */
@ApplicationScoped
public class SelectionPolicy {

    private static final Logger log = Logger.getLogger(SelectionPolicy.class);

    @ConfigProperty(name = "inference.routing.selection-policy", defaultValue = "balanced")
    String policyName;

    @ConfigProperty(name = "inference.routing.policies.balanced.latency-weight", defaultValue = "0.4")
    double balancedLatencyWeight;

    @ConfigProperty(name = "inference.routing.policies.balanced.cost-weight", defaultValue = "0.3")
    double balancedCostWeight;

    @ConfigProperty(name = "inference.routing.policies.balanced.availability-weight", defaultValue = "0.3")
    double balancedAvailabilityWeight;

    /**
     * Rank runners according to configured policy.
     * 
     * @param request    The inference request
     * @param manifest   The model manifest
     * @param candidates List of compatible runners
     * @return Ranked list of runners (best first)
     */
    public List<String> rankRunners(
            InferenceRequest request,
            ModelManifest manifest,
            List<String> candidates) {

        if (candidates.isEmpty()) {
            return Collections.emptyList();
        }

        log.debugf("Ranking %d candidates using %s policy", candidates.size(), policyName);

        // Select policy strategy
        return switch (policyName.toLowerCase()) {
            case "latency" -> rankByLatency(candidates);
            case "cost" -> rankByCost(candidates);
            case "memory" -> rankByMemory(candidates);
            case "balanced" -> rankBalanced(candidates, request);
            default -> {
                log.warnf("Unknown policy %s, using balanced", policyName);
                yield rankBalanced(candidates, request);
            }
        };
    }

    /**
     * Latency-optimized ranking: GPU > TPU > CPU.
     */
    private List<String> rankByLatency(List<String> candidates) {
        return candidates.stream()
                .sorted(Comparator.comparingInt(this::getLatencyScore).reversed())
                .collect(Collectors.toList());
    }

    /**
     * Cost-optimized ranking: CPU > GPU > TPU.
     */
    private List<String> rankByCost(List<String> candidates) {
        return candidates.stream()
                .sorted(Comparator.comparingInt(this::getCostScore).reversed())
                .collect(Collectors.toList());
    }

    /**
     * Memory-optimized ranking: Prefer runners with lower memory usage.
     */
    private List<String> rankByMemory(List<String> candidates) {
        return candidates.stream()
                .sorted(Comparator.comparingInt(this::getMemoryScore).reversed())
                .collect(Collectors.toList());
    }

    /**
     * Balanced ranking: Consider latency, cost, and availability.
     */
    private List<String> rankBalanced(List<String> candidates, InferenceRequest request) {
        Map<String, Double> scores = new HashMap<>();

        for (String runner : candidates) {
            double latencyScore = getLatencyScore(runner) * balancedLatencyWeight;
            double costScore = getCostScore(runner) * balancedCostWeight;
            double availabilityScore = getAvailabilityScore(runner) * balancedAvailabilityWeight;

            double totalScore = latencyScore + costScore + availabilityScore;
            scores.put(runner, totalScore);

            log.debugf("Runner %s scores: latency=%f, cost=%f, availability=%f, total=%f",
                    runner, latencyScore, costScore, availabilityScore, totalScore);
        }

        return candidates.stream()
                .sorted(Comparator.comparingDouble(scores::get).reversed())
                .collect(Collectors.toList());
    }

    /**
     * Calculate latency score (higher is better).
     * GPU/TPU get highest scores.
     */
    private int getLatencyScore(String runnerName) {
        DeviceType device = extractDeviceType(runnerName);

        return switch (device) {
            case CUDA, ROCM, METAL, INTEL_GPU -> 100; // GPU
            case TPU, TPU_V4, TPU_V5 -> 95; // TPU (slightly slower than GPU for small batches)
            case NPU, INFERENTIA, AZURE_MAIA -> 85; // NPU/custom
            case CPU -> 50; // CPU baseline
            default -> 30;
        };
    }

    /**
     * Calculate cost score (higher is better = cheaper).
     * CPU is cheapest.
     */
    private int getCostScore(String runnerName) {
        DeviceType device = extractDeviceType(runnerName);

        return switch (device) {
            case CPU -> 100; // Cheapest
            case NPU -> 80; // Edge devices
            case CUDA, ROCM, METAL -> 50; // GPU moderate cost
            case TPU, TPU_V4, TPU_V5 -> 30; // TPU expensive
            case INFERENTIA, AZURE_MAIA -> 40; // Cloud custom
            default -> 20;
        };
    }

    /**
     * Calculate memory score (higher is better = less memory).
     */
    private int getMemoryScore(String runnerName) {
        // Prefer quantized models and efficient runners
        if (runnerName.contains("int8") || runnerName.contains("quantized")) {
            return 100;
        }

        DeviceType device = extractDeviceType(runnerName);

        return switch (device) {
            case CPU -> 90; // CPU has most available memory
            case NPU -> 80; // Edge optimized
            case CUDA, ROCM -> 60; // GPU limited VRAM
            case TPU -> 70; // TPU moderate
            default -> 50;
        };
    }

    /**
     * Calculate availability score (higher is better).
     * Based on current load and health.
     */
    private int getAvailabilityScore(String runnerName) {
        // In production, this would check:
        // - Current runner load (concurrent requests)
        // - Recent failure rate
        // - Health check status
        // - Queue length

        // For now, return baseline
        return 80;
    }

    /**
     * Extract device type from runner name.
     */
    private DeviceType extractDeviceType(String runnerName) {
        String lower = runnerName.toLowerCase();

        if (lower.contains("cuda") || lower.contains("gpu")) {
            return DeviceType.CUDA;
        } else if (lower.contains("cpu")) {
            return DeviceType.CPU;
        } else if (lower.contains("tpu")) {
            return DeviceType.TPU;
        } else if (lower.contains("npu")) {
            return DeviceType.NPU;
        } else if (lower.contains("metal")) {
            return DeviceType.METAL;
        } else if (lower.contains("rocm")) {
            return DeviceType.ROCM;
        } else if (lower.contains("inferentia")) {
            return DeviceType.INFERENTIA;
        }

        return DeviceType.CPU; // Default
    }

    /**
     * Get human-readable explanation of policy decision.
     */
    public String explainRanking(List<String> rankedRunners) {
        if (rankedRunners.isEmpty()) {
            return "No runners available";
        }

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Policy: %s\n", policyName));
        sb.append("Ranked runners:\n");

        for (int i = 0; i < rankedRunners.size(); i++) {
            String runner = rankedRunners.get(i);
            sb.append(String.format("  %d. %s (latency=%d, cost=%d, memory=%d)\n",
                    i + 1, runner,
                    getLatencyScore(runner),
                    getCostScore(runner),
                    getMemoryScore(runner)));
        }

        return sb.toString();
    }

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

    /**
     * Rank providers based on criteria
     */
    public List<ProviderCandidate> rankProviders(
            ModelManifest manifest,
            RoutingContext context,
            List<LLMProvider> providers) {
        // Implemented in ModelRouterService
        return List.of();
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
        if (manifest.resourceRequirements() == null || manifest.resourceRequirements().memory() == null
                || manifest.resourceRequirements().memory().minMemoryMb() == null) {
            return true;
        }

        long reqMemoryMb = manifest.resourceRequirements().memory().minMemoryMb();
        return (hw.getAvailableMemory() / (1024 * 1024)) >= reqMemoryMb;
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