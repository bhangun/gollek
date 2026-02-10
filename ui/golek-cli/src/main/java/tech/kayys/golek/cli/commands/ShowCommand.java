package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.model.ModelInfo;

import java.util.Optional;

/**
 * Show model details using GolekSdk.
 * Usage: golek show <model-id>
 */
@Dependent
@Unremovable
@Command(name = "show", description = "Show details for a specific model")
public class ShowCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Parameters(index = "0", description = "Model ID to show")
    String modelId;

    @Override
    public void run() {
        try {
            Optional<ModelInfo> modelOpt = sdk.getModelInfo(modelId);

            if (modelOpt.isEmpty()) {
                System.err.println("Model not found: " + modelId);
                return;
            }

            ModelInfo model = modelOpt.get();
            printModelDetails(model);

        } catch (Exception e) {
            System.err.println("Failed to show model: " + e.getMessage());
        }
    }

    private void printModelDetails(ModelInfo model) {
        System.out.println("Model Details");
        System.out.println("=".repeat(50));
        System.out.printf("ID:       %s%n", model.getModelId());
        System.out.printf("Name:     %s%n", model.getName() != null ? model.getName() : "N/A");
        System.out.printf("Version:  %s%n", model.getVersion() != null ? model.getVersion() : "N/A");
        System.out.printf("Format:   %s%n", model.getFormat() != null ? model.getFormat() : "N/A");
        System.out.printf("Size:     %s%n", model.getSizeFormatted());
        if (model.getQuantization() != null) {
            System.out.printf("Quant:    %s%n", model.getQuantization());
        }
        System.out.printf("Created:  %s%n", model.getCreatedAt() != null ? model.getCreatedAt() : "N/A");
        System.out.printf("Modified: %s%n", model.getUpdatedAt() != null ? model.getUpdatedAt() : "N/A");

        if (model.getMetadata() != null && !model.getMetadata().isEmpty()) {
            System.out.println("\nMetadata:");
            model.getMetadata().forEach((key, value) -> System.out.printf("  %s: %s%n", key, value));
        }
    }
}
