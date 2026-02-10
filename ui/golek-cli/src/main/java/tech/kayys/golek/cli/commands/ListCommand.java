package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.model.ModelInfo;

import java.util.List;

/**
 * List local models using GolekSdk.
 * Usage: golek list [--format table|json] [--limit N]
 */
@Dependent
@Unremovable
@Command(name = "list", description = "List available models")
public class ListCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Option(names = { "-f", "--format" }, description = "Output format: table, json", defaultValue = "table")
    String format;

    @Option(names = { "-l", "--limit" }, description = "Maximum models to list", defaultValue = "50")
    int limit;

    @Override
    public void run() {
        try {
            List<ModelInfo> models = sdk.listModels(0, limit);

            if (models.isEmpty()) {
                System.out.println("No models found.");
                return;
            }

            if ("json".equalsIgnoreCase(format)) {
                printJson(models);
            } else {
                printTable(models);
            }
        } catch (Exception e) {
            System.err.println("Failed to list models: " + e.getMessage());
        }
    }

    private void printTable(List<ModelInfo> models) {
        System.out.printf("%-30s %-12s %-10s %-20s%n", "NAME", "SIZE", "FORMAT", "MODIFIED");
        System.out.println("-".repeat(75));

        for (ModelInfo model : models) {
            String modified = model.getUpdatedAt() != null
                    ? model.getUpdatedAt().toString().substring(0, 10)
                    : "N/A";
            System.out.printf("%-30s %-12s %-10s %-20s%n",
                    truncate(model.getModelId(), 30),
                    model.getSizeFormatted(),
                    model.getFormat() != null ? model.getFormat() : "N/A",
                    modified);
        }
        System.out.printf("%n%d model(s) found%n", models.size());
    }

    private void printJson(List<ModelInfo> models) {
        System.out.println("[");
        for (int i = 0; i < models.size(); i++) {
            ModelInfo model = models.get(i);
            System.out.printf("  {\"modelId\": \"%s\", \"name\": \"%s\", \"size\": %d, \"format\": \"%s\"}%s%n",
                    model.getModelId(),
                    model.getName() != null ? model.getName() : model.getModelId(),
                    model.getSizeBytes() != null ? model.getSizeBytes() : 0,
                    model.getFormat() != null ? model.getFormat() : "",
                    i < models.size() - 1 ? "," : "");
        }
        System.out.println("]");
    }

    private String truncate(String str, int maxLen) {
        if (str == null)
            return "";
        return str.length() > maxLen ? str.substring(0, maxLen - 3) + "..." : str;
    }
}
