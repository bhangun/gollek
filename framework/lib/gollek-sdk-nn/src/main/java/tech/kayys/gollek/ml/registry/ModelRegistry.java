package tech.kayys.gollek.ml.registry;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.StateDict;

import java.io.IOException;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Model Registry — versioned storage and retrieval of trained models.
 *
 * <p>Stores model weights (SafeTensors), metadata, and version history.
 * Supports tagging, aliasing (e.g. "production", "staging"), and rollback.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var registry = new ModelRegistry(Path.of("~/.gollek/registry"));
 *
 * // Register a trained model
 * String version = registry.register("sentiment-classifier", model,
 *     Map.of("accuracy", "0.94", "dataset", "imdb"));
 *
 * // Promote to production
 * registry.tag(version, "production");
 *
 * // Load production model
 * Map<String, GradTensor> weights = registry.load("sentiment-classifier", "production");
 * }</pre>
 */
public final class ModelRegistry {

    private final Path rootDir;

    /** In-memory index: modelName → version → ModelEntry */
    private final Map<String, Map<String, ModelEntry>> index = new ConcurrentHashMap<>();

    /**
     * Model registry entry.
     *
     * @param version    unique version string (timestamp-based)
     * @param modelName  model name
     * @param weightsPath path to SafeTensors weights file
     * @param metadata   arbitrary key-value metadata
     * @param createdAt  creation timestamp
     * @param tags       set of tags (e.g. "production", "staging")
     */
    public record ModelEntry(
        String version, String modelName, Path weightsPath,
        Map<String, String> metadata, Instant createdAt, Set<String> tags
    ) {}

    /**
     * Creates a model registry backed by the given directory.
     *
     * @param rootDir root directory for model storage (created if absent)
     * @throws IOException if the directory cannot be created
     */
    public ModelRegistry(Path rootDir) throws IOException {
        this.rootDir = rootDir;
        Files.createDirectories(rootDir);
    }

    /**
     * Registers a trained model and saves its weights.
     *
     * @param name     model name (e.g. "sentiment-classifier")
     * @param model    trained model
     * @param metadata arbitrary metadata (accuracy, dataset, etc.)
     * @return version string for this registration
     * @throws IOException if saving fails
     */
    public String register(String name, NNModule model, Map<String, String> metadata)
            throws IOException {
        String version = name + "-" + System.currentTimeMillis();
        Path modelDir  = rootDir.resolve(name).resolve(version);
        Files.createDirectories(modelDir);

        Path weightsPath = modelDir.resolve("model.safetensors");
        StateDict.save(model.stateDict(), weightsPath);

        // Write metadata
        List<String> lines = new ArrayList<>();
        metadata.forEach((k, v) -> lines.add(k + "=" + v));
        Files.write(modelDir.resolve("metadata.properties"), lines);

        ModelEntry entry = new ModelEntry(version, name, weightsPath,
            Map.copyOf(metadata), Instant.now(), new HashSet<>());
        index.computeIfAbsent(name, k -> new LinkedHashMap<>()).put(version, entry);
        return version;
    }

    /**
     * Tags a model version with a label (e.g. "production", "staging").
     * Only one version per model can hold a given tag at a time.
     *
     * @param version version string returned by {@link #register}
     * @param tag     tag label
     * @throws NoSuchElementException if the version does not exist
     */
    public void tag(String version, String tag) {
        for (Map<String, ModelEntry> versions : index.values()) {
            // Remove tag from any existing holder
            versions.values().forEach(e -> e.tags().remove(tag));
            if (versions.containsKey(version)) {
                versions.get(version).tags().add(tag);
                return;
            }
        }
        throw new NoSuchElementException("Version not found: " + version);
    }

    /**
     * Loads model weights by name and tag (or version string).
     *
     * @param name       model name
     * @param tagOrVersion tag (e.g. "production") or exact version string
     * @return state dict map
     * @throws IOException if loading fails
     * @throws NoSuchElementException if no matching model found
     */
    public Map<String, GradTensor> load(String name, String tagOrVersion) throws IOException {
        Map<String, ModelEntry> versions = index.get(name);
        if (versions == null) throw new NoSuchElementException("Model not found: " + name);

        // Try exact version match first
        ModelEntry entry = versions.get(tagOrVersion);
        if (entry == null) {
            // Try tag match
            entry = versions.values().stream()
                .filter(e -> e.tags().contains(tagOrVersion))
                .findFirst()
                .orElseThrow(() -> new NoSuchElementException(
                    "No version with tag '" + tagOrVersion + "' for model: " + name));
        }
        return StateDict.load(entry.weightsPath());
    }

    /**
     * Lists all registered versions for a model, newest first.
     *
     * @param name model name
     * @return list of {@link ModelEntry} records
     */
    public List<ModelEntry> list(String name) {
        Map<String, ModelEntry> versions = index.getOrDefault(name, Map.of());
        List<ModelEntry> result = new ArrayList<>(versions.values());
        result.sort(Comparator.comparing(ModelEntry::createdAt).reversed());
        return Collections.unmodifiableList(result);
    }

    /**
     * Returns all model names in the registry.
     *
     * @return set of model names
     */
    public Set<String> modelNames() { return Collections.unmodifiableSet(index.keySet()); }
}
