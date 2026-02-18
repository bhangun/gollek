package tech.kayys.gollek.spi.model;

import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Enumeration of supported model formats.
 * 
 * <p>
 * Each format defines:
 * <ul>
 * <li>Unique identifier for lookups</li>
 * <li>Display name for UI presentation</li>
 * <li>File extensions associated with the format</li>
 * <li>Marker files that indicate the format</li>
 * <li>Whether conversion to GGUF is supported</li>
 * <li>Primary runtime engine</li>
 * </ul>
 */
public enum ModelFormat {

    // Convertible formats (can be converted to GGUF)
    PYTORCH("pytorch", "PyTorch",
            Set.of(".bin", ".pt", ".pth"),
            Set.of("pytorch_model.bin", "model.safetensors"),
            true,
            "PyTorch"),

    SAFETENSORS("safetensors", "SafeTensors",
            Set.of(".safetensors"),
            Set.of("model.safetensors"),
            true,
            "SafeTensors"),

    TENSORFLOW("tensorflow", "TensorFlow",
            Set.of(".pb", ".h5"),
            Set.of("saved_model.pb", "tf_model.h5"),
            true,
            "TensorFlow"),

    FLAX("flax", "Flax/JAX",
            Set.of(".msgpack"),
            Set.of("flax_model.msgpack"),
            true,
            "JAX"),

    // Native inference formats (no conversion needed)
    GGUF("gguf", "GGUF",
            Set.of(".gguf"),
            Set.of(),
            false,
            "llama.cpp"),

    LITERT("tflite", "LiteRT",
            Set.of(".tflite"),
            Set.of(),
            false,
            "LiteRT"),

    ONNX("onnx", "ONNX",
            Set.of(".onnx"),
            Set.of("model.onnx"),
            false,
            "ONNX Runtime"),

    TENSORRT("trt", "TensorRT",
            Set.of(".trt", ".engine"),
            Set.of(),
            false,
            "TensorRT"),

    TORCHSCRIPT("torchscript", "TorchScript",
            Set.of(".pt", ".pts"),
            Set.of(),
            false,
            "PyTorch"),

    TENSORFLOW_SAVED_MODEL("pb", "TensorFlow SavedModel",
            Set.of(".pb"),
            Set.of("saved_model.pb"),
            false,
            "TensorFlow"),

    UNKNOWN("unknown", "Unknown",
            Set.of(),
            Set.of(),
            false,
            "Unknown");

    private final String id;
    private final String displayName;
    private final Set<String> fileExtensions;
    private final Set<String> markerFiles;
    private final boolean requiresConversion;
    private final String runtime;

    /**
     * Constructor for model format.
     * 
     * @param id                 unique identifier
     * @param displayName        human-readable name
     * @param fileExtensions     set of file extensions (with leading dot)
     * @param markerFiles        set of marker filenames that indicate this format
     * @param requiresConversion whether this format can be converted to GGUF
     * @param runtime            primary runtime engine name
     */
    ModelFormat(String id, String displayName,
            Set<String> fileExtensions, Set<String> markerFiles,
            boolean requiresConversion, String runtime) {
        this.id = id;
        this.displayName = displayName;
        this.fileExtensions = Collections.unmodifiableSet(fileExtensions);
        this.markerFiles = Collections.unmodifiableSet(markerFiles);
        this.requiresConversion = requiresConversion;
        this.runtime = runtime;
    }

    // Getters

    /**
     * Gets the unique identifier for this format.
     * 
     * @return format ID
     */
    public String getId() {
        return id;
    }

    /**
     * Gets the display name for UI presentation.
     * 
     * @return display name
     */
    public String getDisplayName() {
        return displayName;
    }

    /**
     * Gets the file extensions associated with this format.
     * 
     * @return unmodifiable set of extensions (with leading dot)
     */
    public Set<String> getFileExtensions() {
        return fileExtensions;
    }

    /**
     * Gets the primary file extension (first in the set).
     * 
     * @return primary extension or empty string if none
     */
    public String getExtension() {
        return fileExtensions.isEmpty() ? "" : fileExtensions.iterator().next();
    }

    /**
     * Gets marker files that indicate this format.
     * 
     * @return unmodifiable set of marker filenames
     */
    public Set<String> getMarkerFiles() {
        return markerFiles;
    }

    /**
     * Gets the runtime engine name.
     * 
     * @return runtime name
     */
    public String getRuntime() {
        return runtime;
    }

    /**
     * Checks if this format requires conversion to GGUF.
     * 
     * @return true if conversion is supported
     */
    public boolean isRequiresConversion() {
        return requiresConversion;
    }

    /**
     * Checks if this format requires conversion to GGUF.
     * Alias for {@link #isRequiresConversion()} for backward compatibility.
     * 
     * @return true if conversion is supported
     */
    public boolean requiresConversion() {
        return requiresConversion;
    }

    /**
     * Checks if this format can be converted to GGUF.
     * Alias for {@link #isRequiresConversion()} with additional validation.
     * 
     * @return true if can be converted to GGUF
     */
    public boolean isConvertible() {
        return requiresConversion && this != UNKNOWN;
    }

    // Static lookup methods

    /**
     * Finds a format by its unique ID.
     * 
     * @param id format ID (case-insensitive)
     * @return matching format or UNKNOWN if not found
     */
    public static ModelFormat fromId(String id) {
        if (id == null || id.isBlank()) {
            return UNKNOWN;
        }

        String normalized = id.toLowerCase().trim();
        return Arrays.stream(values())
                .filter(f -> f.id.equals(normalized))
                .findFirst()
                .orElse(UNKNOWN);
    }

    /**
     * Finds a format by file extension.
     * 
     * @param extension file extension (with or without leading dot,
     *                  case-insensitive)
     * @return matching format or UNKNOWN if not found
     */
    public static ModelFormat fromExtension(String extension) {
        if (extension == null || extension.isBlank()) {
            return UNKNOWN;
        }

        String normalized = extension.toLowerCase().trim();
        if (!normalized.startsWith(".")) {
            normalized = "." + normalized;
        }

        final String ext = normalized;
        return Arrays.stream(values())
                .filter(f -> f.fileExtensions.contains(ext))
                .findFirst()
                .orElse(UNKNOWN);
    }

    /**
     * Finds a format by file extension.
     * Alias for {@link #fromExtension(String)} that returns Optional for backward
     * compatibility.
     * 
     * @param extension file extension (with or without leading dot,
     *                  case-insensitive)
     * @return Optional containing the format if found, empty if UNKNOWN
     */
    public static Optional<ModelFormat> findByExtension(String extension) {
        ModelFormat format = fromExtension(extension);
        return format == UNKNOWN ? Optional.empty() : Optional.of(format);
    }

    /**
     * Finds a format by runtime engine name.
     * 
     * @param runtime runtime name (case-insensitive)
     * @return Optional containing the format if found
     */
    public static Optional<ModelFormat> findByRuntime(String runtime) {
        if (runtime == null || runtime.isBlank()) {
            return Optional.empty();
        }

        return Arrays.stream(values())
                .filter(format -> format.runtime.equalsIgnoreCase(runtime.trim()))
                .findFirst();
    }

    /**
     * Finds a format by marker filename.
     * 
     * @param filename marker filename to search for
     * @return matching format or UNKNOWN if not found
     */
    public static ModelFormat fromMarkerFile(String filename) {
        if (filename == null || filename.isBlank()) {
            return UNKNOWN;
        }

        String normalized = filename.toLowerCase().trim();
        return Arrays.stream(values())
                .filter(f -> f.markerFiles.stream()
                        .anyMatch(marker -> marker.equalsIgnoreCase(normalized)))
                .findFirst()
                .orElse(UNKNOWN);
    }

    /**
     * Gets all formats that support conversion to GGUF.
     * 
     * @return unmodifiable set of convertible formats
     */
    public static Set<ModelFormat> getConvertibleFormats() {
        return Arrays.stream(values())
                .filter(ModelFormat::isRequiresConversion)
                .collect(Collectors.toUnmodifiableSet());
    }

    /**
     * Gets all native inference formats (no conversion needed).
     * 
     * @return unmodifiable set of native formats
     */
    public static Set<ModelFormat> getNativeFormats() {
        return Arrays.stream(values())
                .filter(f -> !f.requiresConversion && f != UNKNOWN)
                .collect(Collectors.toUnmodifiableSet());
    }

    /**
     * Checks if a filename matches this format based on extension.
     * 
     * @param filename filename to check
     * @return true if filename has a matching extension
     */
    public boolean matches(String filename) {
        if (filename == null || filename.isBlank()) {
            return false;
        }

        String lower = filename.toLowerCase();
        return fileExtensions.stream()
                .anyMatch(lower::endsWith);
    }

    @Override
    public String toString() {
        return String.format("%s (%s) - Runtime: %s, Convertible: %s",
                displayName, id, runtime, requiresConversion);
    }
}
