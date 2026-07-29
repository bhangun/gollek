package tech.kayys.gollek.spi.registry;

import tech.kayys.alkhawarizm.core.model.ModelFormat;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Local model registry for managing locally stored models.
 *
 * @since 2.1.0
 */
public interface LocalModelRegistry {

    /**
     * Register a model in the registry.
     */
    ModelEntry register(String modelId, Path physicalPath, ModelFormat format);

    /**
     * Add directory roots to scan for model files.
     */
    void addScanRoots(Path... paths);

    /**
     * Register an alias for a model.
     */
    void registerAlias(String alias, String modelId);

    /**
     * Resolve model identifier or path to an entry.
     */
    Optional<ModelEntry> resolve(String modelRef);

    /**
     * Resolve all matching model entries for the given identifier.
     * Useful when multiple formats exist for the same model name.
     *
     * @param modelRef model identifier or path
     * @return list of matching entries
     */
    List<ModelEntry> resolveAll(String modelRef);

    /**
     * List all registered entries.
     */
    List<ModelEntry> listAll(ModelFormat format);

    /**
     * Refresh the registry by scanning roots.
     */
    void refresh();

    /**
     * Clear the registry.
     */
    void clear();

    /**
     * Load registry from file.
     * 
     * @param path Path to registry file
     */
    void load(Path path);

    /**
     * Save registry to file.
     *
     * @param path Path to registry file
     */
    void save(Path path);

    /** Marker method to force re-indexing. */
    default void touch() {
    }
}
