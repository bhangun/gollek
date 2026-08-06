package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelRegistry.ModelStats;

import java.util.Map;
import java.util.Optional;

@Dependent
@Unremovable
@Command(name = "stats", description = "Display detailed metrics and statistics for a model/provider")
public class StatsCommand implements Runnable {

    @Inject
    GollekSdk sdk;

    @Option(names = { "-m", "--model" }, description = "Model ID", required = true)
    public String modelId;

    @Option(names = { "--provider" }, description = "Provider ID for metrics", defaultValue = "gguf")
    public String providerId;

    @Override
    public void run() {
        try {
            System.out.println("Retrieving statistics for model: " + modelId);
            Optional<ModelStats> statsOpt = sdk.getModelStats(modelId);
            if (statsOpt.isPresent()) {
                ModelStats stats = statsOpt.get();
                System.out.println("Total Inferences: " + stats.totalInferences());
                System.out.println("Version Count: " + stats.versionCount());
                System.out.println("Stage: " + stats.stage());
            } else {
                System.out.println("No generic statistics available for this model.");
            }

            System.out.println("\n--- Provider Metrics (" + providerId + ") ---");
            Map<String, Object> metrics = sdk.getMetrics(providerId, modelId);
            if (metrics.isEmpty()) {
                System.out.println("No runtime metrics available or unsupported by provider.");
            } else {
                metrics.forEach((k, v) -> System.out.println(k + ": " + v));
            }
            
        } catch (Exception e) {
            System.err.println("Failed to fetch stats: " + e.getMessage());
        }
    }
}
