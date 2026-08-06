package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelManifest;
import tech.kayys.alkhawarizm.spi.model.ModelRegistry.ModelUploadRequest;

@Dependent
@Unremovable
@Command(name = "register", description = "Register a remote or local model into the internal cache")
public class RegisterCommand implements Runnable {

    @Inject
    GollekSdk sdk;

    @Option(names = { "-n", "--name" }, description = "Name to register the model as", required = true)
    public String name;

    @Option(names = { "-u", "--url" }, description = "Source URL or local path for the model", required = true)
    public String url;

    @Option(names = { "--format" }, description = "Model format (e.g., gguf, safetensors)", defaultValue = "gguf")
    public String format;

    @Override
    public void run() {
        try {
            System.out.printf("Registering model '%s' from '%s' [%s]%n", name, url, format);
            
            ModelUploadRequest request = new ModelUploadRequest(null, name, "1.0.0", name, "Registered from CLI", format, null, new String[]{}, java.util.Map.of("url", url), java.util.Map.of(), java.util.Map.of(), "cli-user");
            ModelManifest manifest = sdk.registerModel(request);
            
            System.out.println("Successfully registered model.");
            System.out.println("ID: " + manifest.modelId());
            System.out.println("Created: " + manifest.createdAt());
            
        } catch (UnsupportedOperationException e) {
            System.err.println("Register operation is not supported by the current SDK implementation.");
        } catch (Exception e) {
            System.err.println("Failed to register model: " + e.getMessage());
        }
    }
}
