package tech.kayys.gollek.spi.registry;

import tech.kayys.gollek.spi.model.ModelFormat;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * SPI contract for a local model file catalogue.
 *
 * <p>Implementations discover, register, and resolve locally-available model
 * files (GGUF, SafeTensors, etc.) by scanning configured base directories.
 *
 * <p><strong>This is NOT the same as {@link tech.kayys.gollek.spi.model.ModelRegistry}</strong>,
 * which manages model manifests, tenants, and persistence.  {@code LocalModelRegistry}
 * is concerned only with <em>file-level</em> discovery and resolution.
 *
 * <h3>Resolution order (contract)</h3>
 * <ol>
 *   <li>Exact index hit — previously registered by full path or explicit id.</li>
 *   <li>Alias map lookup — e.g. {@code "llama3"} → canonical id.</li>
 *   <li>Absolute path — if {@code modelRef} is a valid file path.</li>
 *   <li>On-demand scan — all configured base directories are walked (depth 3).</li>
 *   <li>Fuzzy stem match — case-insensitive filename without extension.</li>
 * </ol>
 *
 * <p>Implementations must be thread-safe.
 */
public interface LocalModelRegistry {

    /** Immutable snapshot of a registered model file. */
    record ModelEntry(
            String modelId,
            Path physicalPath,
            ModelFormat format,
            Instant registeredAt) {

        /** Short display name derived from the file or parent directory. */
        public String displayName() {
            String name = physicalPath.getFileName().toString();
            // For SafeTensors: prefer the parent dir name (it's the "model folder")
            if (format == ModelFormat.SAFETENSORS) {
                Path parent = physicalPath.getParent();
                if (parent != null && parent.getFileName() != null) {
                    return parent.getFileName().toString();
                }
            }
            return name;
        }
    }

    // ── Registration ──────────────────────────────────────────────────────────

    /**
     * Register base paths to scan.  Called during provider initialisation.
     *
     * @param paths one or more directories to include in discovery
     */
    void addScanRoots(Path... paths);

    /**
     * Explicitly register a model file.  Overwrites any existing entry for the
     * same {@code modelId}.
     *
     * @param modelId      logical identifier (may be a path or friendly name)
     * @param physicalPath absolute path to the model file
     * @param format       format hint; if {@code null} format is auto-detected
     * @return the registered entry
     */
    ModelEntry register(String modelId, Path physicalPath, ModelFormat format);

    /**
     * Register a friendly alias for an already-known {@code modelId}.
     *
     * @param alias   e.g. {@code "llama3"}
     * @param modelId canonical key already present in the index
     */
    void registerAlias(String alias, String modelId);

    // ── Lookup ────────────────────────────────────────────────────────────────

    /**
     * Resolve a model reference to a registry entry.
     *
     * @param modelRef path, alias, or friendly name
     * @return the entry, or {@link Optional#empty()} when not found
     */
    Optional<ModelEntry> resolve(String modelRef);

    /**
     * Return all registered entries, optionally filtered by format.
     *
     * @param format format filter; {@code null} = return all
     * @return unmodifiable snapshot
     */
    List<ModelEntry> listAll(ModelFormat format);

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /** Refresh the index by re-scanning all registered base paths. */
    void refresh();

    /** Remove all entries (useful in tests). */
    void clear();
}
