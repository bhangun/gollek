package tech.kayys.gollek.provider.core.startup;

import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.registry.LocalModelRegistry;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Validates provider configuration at startup.
 *
 * <p>
 * Emits clear {@code WARN} messages (not crashes) for misconfigured base
 * paths so operators know why no models are discovered. Also logs a
 * human-readable summary of all models found in the registry after the initial
 * scan.
 *
 * <p>
 * Uses {@code @ConfigProperty} to read provider base-path and enabled flags
 * directly, avoiding a compile-time dependency on extension-module config
 * interfaces ({@code GGUFProviderConfig}, {@code SafetensorProviderConfig}).
 */
@ApplicationScoped
public class StartupValidator {

    private static final Logger LOG = Logger.getLogger(StartupValidator.class);

    @ConfigProperty(name = "gguf.provider.enabled", defaultValue = "true")
    boolean ggufEnabled;

    @ConfigProperty(name = "gguf.provider.model.base-path", defaultValue = "${user.home}/.gollek/models/gguf")
    String ggufBasePath;

    @ConfigProperty(name = "safetensor.provider.enabled", defaultValue = "true")
    boolean safetensorEnabled;

    @ConfigProperty(name = "safetensor.provider.basePath", defaultValue = "${user.home}/.gollek/models/safetensors")
    String safetensorBasePath;

    @Inject
    LocalModelRegistry localModelRegistry;

    void onStart(@Observes StartupEvent ev) {
        List<String> warnings = new ArrayList<>();

        // ── GGUF ──────────────────────────────────────────────────────────────
        if (ggufEnabled) {
            warnings.addAll(validatePath("GGUF", ggufBasePath));
        } else {
            LOG.info("StartupValidator: GGUF provider is disabled");
        }

        // ── SafeTensors ───────────────────────────────────────────────────────
        if (safetensorEnabled) {
            warnings.addAll(validatePath("SafeTensors", safetensorBasePath));
        } else {
            LOG.info("StartupValidator: SafeTensors provider is disabled");
        }

        if (warnings.isEmpty()) {
            LOG.info("StartupValidator: all provider base paths look correct");
        } else {
            warnings.forEach(LOG::warn);
        }

        // Trigger an initial registry scan so models are available immediately
        localModelRegistry.refresh();
        logRegistrySummary();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private List<String> validatePath(String providerName, String rawPath) {
        List<String> issues = new ArrayList<>();
        if (rawPath == null || rawPath.isBlank()) {
            issues.add("StartupValidator [" + providerName + "]: basePath is empty — no models will be discovered");
            return issues;
        }
        try {
            Path p = Path.of(rawPath);
            if (!Files.exists(p)) {
                issues.add("StartupValidator [" + providerName + "]: basePath does not exist — "
                        + rawPath + " — create it and place model files there");
            } else if (!Files.isDirectory(p)) {
                issues.add("StartupValidator [" + providerName + "]: basePath is not a directory — " + rawPath);
            } else if (!Files.isReadable(p)) {
                issues.add("StartupValidator [" + providerName + "]: basePath is not readable — " + rawPath
                        + " — check file system permissions");
            } else {
                LOG.infof("StartupValidator [%s]: basePath OK — %s", providerName, rawPath);
            }
        } catch (Exception e) {
            issues.add("StartupValidator [" + providerName + "]: cannot evaluate basePath '"
                    + rawPath + "' — " + e.getMessage());
        }
        return issues;
    }

    private void logRegistrySummary() {
        var ggufModels = localModelRegistry.listAll(ModelFormat.GGUF);
        var safetensorModels = localModelRegistry.listAll(ModelFormat.SAFETENSORS);

        LOG.infof("LocalModelRegistry summary: GGUF=%d, SafeTensors=%d model(s) discovered",
                ggufModels.size(), safetensorModels.size());

        if (LOG.isDebugEnabled()) {
            ggufModels.forEach(e -> LOG.debugf("  [GGUF]        %s → %s", e.displayName(), e.physicalPath()));
            safetensorModels.forEach(e -> LOG.debugf("  [SAFETENSORS] %s → %s", e.displayName(), e.physicalPath()));
        }

        if (ggufModels.isEmpty() && safetensorModels.isEmpty()) {
            LOG.warn("StartupValidator: no local models discovered — "
                    + "place .gguf or .safetensors files in the configured base paths");
        }
    }
}
