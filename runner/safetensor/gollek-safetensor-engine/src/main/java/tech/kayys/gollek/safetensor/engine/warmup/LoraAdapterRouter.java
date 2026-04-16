/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * LoraAdapterRouter.java
 * ──────────────────────
 * Resolves and routes per-request LoRA adapters for multi-tenant deployments.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.Config;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-request LoRA adapter resolver for multi-tenant inference.
 */
@ApplicationScoped
public class LoraAdapterRouter {

    private static final Logger log = Logger.getLogger(LoraAdapterRouter.class);

    /** Prefix for adapter alias → path mappings in application.properties. */
    private static final String CONFIG_PREFIX = "gollek.safetensor.adapters.";

    @Inject
    LoraAdapterService loraAdapter;
    @Inject
    Config mpConfig;

    /**
     * In-memory alias → path registry.
     */
    private final Map<String, Path> registry = new ConcurrentHashMap<>();

    // ─────────────────────────────────────────────────────────────────────────
    // Registry management
    // ─────────────────────────────────────────────────────────────────────────

    public void register(String alias, Path adapterDir) {
        if (!Files.isDirectory(adapterDir)) {
            log.warnf("LoraAdapterRouter: adapter path '%s' for alias '%s' is not a directory",
                    adapterDir, alias);
        }
        registry.put(alias, adapterDir);
        log.infof("LoraAdapterRouter: registered adapter '%s' → %s", alias, adapterDir);
    }

    public void unregister(String alias) {
        registry.remove(alias);
    }

    public Map<String, Path> registeredAdapters() {
        return Map.copyOf(registry);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Resolution
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Resolve the LoRA adapter for an incoming request.
     */
    public Optional<LoadedAdapter> resolve(ProviderRequest request) {
        Optional<Path> adapterPath = findAdapterPath(request);
        if (adapterPath.isEmpty())
            return Optional.empty();

        Path path = adapterPath.get();
        try {
            LoadedAdapter loaded = loraAdapter.load(path);
            log.debugf("LoraAdapterRouter: resolved adapter for request %s → %s",
                    request.getRequestId(), path.getFileName());
            return Optional.of(loaded);
        } catch (Exception e) {
            log.warnf(e, "LoraAdapterRouter: failed to load adapter from %s for request %s",
                    path, request.getRequestId());
            return Optional.empty();
        }
    }

    public Optional<Path> resolveAdapterPath(ProviderRequest request) {
        return findAdapterPath(request);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private Optional<Path> findAdapterPath(ProviderRequest request) {
        Map<String, Object> params = request.getParameters();
        Map<String, Object> metadata = request.getMetadata();

        // 1. adapter_id from request parameters
        String adapterId = asString(params.get("adapter_id"));
        if (adapterId == null)
            adapterId = asString(params.get("lora_adapter_id"));
        if (adapterId == null)
            adapterId = asString(params.get("lora_adapter"));
        if (adapterId == null)
            adapterId = asString(metadata.get("adapter_id"));

        if (adapterId != null) {
            Path registered = registry.get(adapterId);
            if (registered != null)
                return Optional.of(registered);
            // Maybe it's a path directly
            Path direct = Path.of(adapterId);
            if (Files.isDirectory(direct))
                return Optional.of(direct);
            // Try loading from config
            Path fromConfig = loadFromConfig(adapterId);
            if (fromConfig != null) {
                register(adapterId, fromConfig);
                return Optional.of(fromConfig);
            }
            log.warnf("LoraAdapterRouter: adapter_id '%s' not found in registry", adapterId);
        }

        // 2. adapter_path from request parameters (direct path)
        String adapterPath = asString(params.get("adapter_path"));
        if (adapterPath == null)
            adapterPath = asString(params.get("lora_adapter_path"));
        if (adapterPath == null)
            adapterPath = asString(metadata.get("adapter_path"));

        if (adapterPath != null) {
            Path p = Path.of(adapterPath);
            if (Files.isDirectory(p))
                return Optional.of(p);
            log.warnf("LoraAdapterRouter: adapter_path '%s' is not a directory", adapterPath);
        }

        return Optional.empty();
    }

    private Path loadFromConfig(String alias) {
        return mpConfig.getOptionalValue(CONFIG_PREFIX + alias, String.class)
                .map(Path::of)
                .filter(Files::isDirectory)
                .orElse(null);
    }

    private static String asString(Object o) {
        if (o instanceof String s && !s.isBlank())
            return s.strip();
        return null;
    }
}
