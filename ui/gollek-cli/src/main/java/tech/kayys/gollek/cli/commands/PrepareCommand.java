package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.sdk.local.LocalGollekSdk;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;

@Dependent
@Unremovable
@Command(name = "prepare", description = "Download and prepare a model for inference (including GGUF conversion)")
public class PrepareCommand implements Runnable {

    @Inject
    LocalGollekSdk sdk;

    @Parameters(index = "0", description = "Model name or path (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    String modelName;

    @Option(names = {"--force-gguf"}, description = "Force conversion to GGUF format even if not explicitly requested")
    boolean forceGguf;

    @Option(names = {"--quantization", "-q"}, description = "Quantization type (F32, F16, Q8_0, Q4_0, Q4_K, etc.)", defaultValue = "Q4_K")
    String quantization;

    @Override
    public void run() {
        System.out.println("Preparing model: " + modelName);
        
        try {
            ModelResolution resolution = sdk.prepareModel(modelName, forceGguf, quantization, progress -> {
                if (progress.getTotal() > 0) {
                    int percent = progress.getPercentComplete();
                    System.out.printf("\r[%-50s] %d%% - %s", 
                        "=".repeat(percent / 2), percent, progress.getStatus());
                } else {
                    System.out.printf("\r%s...                      ", progress.getStatus());
                }
                
                if (progress.isComplete()) {
                    System.out.println();
                }
            });

            if (resolution != null && resolution.getLocalPath() != null) {
                System.out.println("\nModel ready at: " + resolution.getLocalPath());
                if (resolution.getInfo() != null) {
                    System.out.println("Format: " + resolution.getInfo().getFormat());
                }
            } else {
                System.err.println("\nFailed to prepare model (no local path resolved).");
            }
        } catch (Exception e) {
            System.err.println("\nError preparing model: " + e.getMessage());
            // log.error("Prepare failed", e);
        }
    }
}
