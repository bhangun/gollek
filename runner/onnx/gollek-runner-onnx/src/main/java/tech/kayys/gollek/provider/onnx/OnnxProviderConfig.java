package tech.kayys.gollek.provider.onnx;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;

@ApplicationScoped
public class OnnxProviderConfig {

    @ConfigProperty(name = "gollek.providers.onnx.enabled", defaultValue = "true")
    boolean enabled;

    @ConfigProperty(name = "gollek.providers.onnx.model-base-path", defaultValue = "${user.home}/.gollek/models")
    String modelBasePath;

    @ConfigProperty(name = "gollek.providers.onnx.threads", defaultValue = "4")
    int threads;

    public boolean enabled() {
        return enabled;
    }

    public String modelBasePath() {
        // Resolve ${user.home} manually if needed, though MicroProfile Config often handles it
        return modelBasePath.replace("${user.home}", System.getProperty("user.home"));
    }

    public int threads() {
        return threads;
    }
}
