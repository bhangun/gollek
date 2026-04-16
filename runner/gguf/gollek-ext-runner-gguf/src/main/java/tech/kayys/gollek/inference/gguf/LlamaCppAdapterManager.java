package tech.kayys.gollek.inference.gguf;

import org.jboss.logging.Logger;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Map;

/**
 * Manages LoRA adapter lifecycle including loading, activation, scaling, and cleanup.
 */
public class LlamaCppAdapterManager {

    private static final Logger log = Logger.getLogger(LlamaCppAdapterManager.class);

    private final LlamaCppBinding binding;

    private MemorySegment activeAdapter;
    private String activeAdapterId;
    private String activeAdapterPath;
    private float activeAdapterScale = 1.0f;

    public LlamaCppAdapterManager(LlamaCppBinding binding) {
        this.binding = binding;
    }

    /**
     * Configure and load a LoRA adapter if specified in runner config.
     */
    public void configureAdapter(MemorySegment model, MemorySegment context, Map<String, Object> runnerConfig) {
        String adapterType = String.valueOf(runnerConfig.getOrDefault("adapter.type", "lora"));
        if (!"lora".equalsIgnoreCase(adapterType)) {
            throw new IllegalArgumentException("GGUF runner only supports adapter.type=lora");
        }

        Object rawPath = runnerConfig.get("adapter.path");
        if (rawPath == null) {
            return;
        }

        String adapterPath = rawPath.toString().trim();
        if (adapterPath.isBlank()) {
            return;
        }

        Path path = Path.of(adapterPath);
        if (!path.isAbsolute()) {
            path = path.toAbsolutePath().normalize();
        }
        if (!path.toFile().exists()) {
            throw new RuntimeException("Adapter file not found: " + path);
        }

        String adapterId = String.valueOf(runnerConfig.getOrDefault("adapter.id", path.getFileName().toString()));
        float adapterScale = getFloatConfig(runnerConfig, "adapter.scale", 1.0f);

        loadAdapter(model, context, path.toString(), adapterId, adapterScale);
    }

    /**
     * Load a LoRA adapter and set it as active.
     */
    public void loadAdapter(MemorySegment model, MemorySegment context, String adapterPath,
                           String adapterId, float adapterScale) {
        MemorySegment handle = null;
        try {
            handle = binding.loadLoraAdapter(model, adapterPath);
            binding.setLoraAdapter(context, handle, adapterScale);

            this.activeAdapter = handle;
            this.activeAdapterId = adapterId;
            this.activeAdapterPath = adapterPath;
            this.activeAdapterScale = adapterScale;

            log.infof("Loaded LoRA adapter %s from %s with scale %.3f",
                    adapterId, adapterPath, adapterScale);
        } catch (Exception e) {
            if (handle != null && !handle.equals(MemorySegment.NULL)) {
                try {
                    binding.freeLoraAdapter(handle);
                } catch (Exception ignored) {
                    // ignore cleanup failures
                }
            }
            throw new RuntimeException("Failed to load adapter " + adapterId, e);
        }
    }

    /**
     * Get the active adapter handle.
     */
    public MemorySegment getActiveAdapter() {
        return activeAdapter;
    }

    /**
     * Get the active adapter ID.
     */
    public String getActiveAdapterId() {
        return activeAdapterId;
    }

    /**
     * Get the active adapter path.
     */
    public String getActiveAdapterPath() {
        return activeAdapterPath;
    }

    /**
     * Get the active adapter scale.
     */
    public float getActiveAdapterScale() {
        return activeAdapterScale;
    }

    /**
     * Check if an adapter is currently loaded.
     */
    public boolean hasActiveAdapter() {
        return activeAdapter != null && !activeAdapter.equals(MemorySegment.NULL);
    }

    /**
     * Remove the active adapter from the context.
     */
    public void removeAdapter(MemorySegment context) {
        if (context != null && activeAdapter != null && !activeAdapter.equals(MemorySegment.NULL)) {
            try {
                binding.removeLoraAdapter(context, activeAdapter);
            } catch (Exception e) {
                log.debugf("Failed to remove adapter %s: %s", activeAdapterId, e.getMessage());
            }
        }
    }

    /**
     * Clear all adapters from the context.
     */
    public void clearAdapters(MemorySegment context) {
        if (context != null) {
            try {
                binding.clearLoraAdapters(context);
            } catch (Exception e) {
                log.debugf("Failed to clear adapters from context: %s", e.getMessage());
            }
        }
    }

    /**
     * Cleanup and free the active adapter resources.
     */
    public void cleanup() {
        if (activeAdapter != null && !activeAdapter.equals(MemorySegment.NULL)) {
            try {
                binding.freeLoraAdapter(activeAdapter);
            } catch (Exception e) {
                log.debugf("Failed to free adapter %s (%s): %s",
                        activeAdapterId, activeAdapterPath, e.getMessage());
            }
        }
        activeAdapter = null;
        activeAdapterId = null;
        activeAdapterPath = null;
        activeAdapterScale = 1.0f;
    }

    private float getFloatConfig(Map<String, Object> config, String key, float defaultValue) {
        if (config == null || !config.containsKey(key)) {
            return defaultValue;
        }
        Object value = config.get(key);
        if (value instanceof Number number) {
            return number.floatValue();
        }
        if (value instanceof String s) {
            try {
                return Float.parseFloat(s.trim());
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        return defaultValue;
    }
}
