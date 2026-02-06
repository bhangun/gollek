package tech.kayys.golek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.golek.sdk.core.GolekSdk;

/**
 * Pull a model from registry using GolekSdk.
 * Usage: golek pull <model-spec>
 * 
 * Model spec formats:
 * - ollama:llama2 -> Pull from Ollama
 * - hf:TheBloke/Llama-2 -> Pull from HuggingFace (future)
 * - llama2 -> Default to Ollama
 */
@Command(name = "pull", description = "Pull a model from registry")
@Dependent
public class PullCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Parameters(index = "0", description = "Model specification (e.g., ollama:llama2, llama2)")
    String modelSpec;

    @Option(names = { "--insecure" }, description = "Allow insecure connections")
    boolean insecure;

    @Override
    public void run() {
        try {
            System.out.println("Pulling model: " + modelSpec);
            System.out.println();

            sdk.pullModel(modelSpec, progress -> {
                if (progress.getTotal() > 0) {
                    String bar = progress.getProgressBar(30);
                    System.out.printf("\r%s [%s] %3d%%",
                            progress.getStatus(),
                            bar,
                            progress.getPercentComplete());
                } else {
                    System.out.printf("\r%s...", progress.getStatus());
                }
            });

            System.out.println("\nPull complete: " + modelSpec);

        } catch (Exception e) {
            System.err.println("\nFailed to pull model: " + e.getMessage());
        }
    }
}
