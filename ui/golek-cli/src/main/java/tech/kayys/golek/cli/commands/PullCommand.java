package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.golek.model.repo.hf.HuggingFaceClient;
import tech.kayys.golek.sdk.core.GolekSdk;

/**
 * Pull model using GolekSdk.
 * Usage: golek pull <model-spec>
 * Model spec examples:
 * - ollama:llama2 -> Pull from Ollama
 * - hf:TheBloke/Llama-2 -> Pull from HuggingFace (future)
 * - llama2 -> Default to Ollama
 */
@Dependent
@Unremovable
@Command(name = "pull", description = "Pull a model from a provider")
public class PullCommand implements Runnable {

    @Inject
    GolekSdk sdk;
    @Inject
    Instance<HuggingFaceClient> hfClientInstance;

    @Parameters(index = "0", description = "Model specification (e.g., ollama:llama2, llama2)")
    String modelSpec;

    @Option(names = { "--insecure" }, description = "Allow insecure connections")
    boolean insecure;

    @Option(names = { "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    @Override
    public void run() {
        try {
            configureCheckpointConversionPreference();
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
            String reason = describeError(e);
            if (HuggingFaceCheckpointStore.shouldStoreOnPullFailure(reason)) {
                var stored = HuggingFaceCheckpointStore.storeCheckpointArtifacts(
                        hfClientInstance,
                        modelSpec,
                        progress -> System.out.printf("\r%s...", progress.getStatus()));
                if (stored.isPresent() && stored.get().hasWeights()) {
                    System.out.println("\nCheckpoint artifacts stored: " + stored.get().rootDir().toAbsolutePath());
                    System.out.println(
                            "Model is stored in origin checkpoint format and not directly runnable in local Java runtime.");
                    return;
                }
            }
            System.err.println("\nFailed to pull model: " + e.getMessage());
            System.err.println("Detail: " + reason);
        }
    }

    private void configureCheckpointConversionPreference() {
        String mode = convertMode == null ? "auto" : convertMode.trim().toLowerCase();
        if (mode.equals("off")) {
            System.setProperty("golek.gguf.converter.auto", "false");
        } else {
            System.setProperty("golek.gguf.converter.auto", "true");
        }
        if (ggufOutType != null && !ggufOutType.isBlank()) {
            System.setProperty("golek.gguf.converter.outtype", ggufOutType.trim().toLowerCase());
        }
    }

    private String describeError(Throwable throwable) {
        if (throwable == null) {
            return "unknown error";
        }
        StringBuilder sb = new StringBuilder();
        Throwable current = throwable;
        int guard = 0;
        while (current != null && guard++ < 8) {
            String msg = current.getMessage();
            if (msg != null && !msg.isBlank()) {
                if (sb.length() > 0) {
                    sb.append(" | ");
                }
                sb.append(msg);
            }
            current = current.getCause();
        }
        if (sb.length() == 0) {
            return throwable.getClass().getSimpleName();
        }
        return sb.toString();
    }
}
