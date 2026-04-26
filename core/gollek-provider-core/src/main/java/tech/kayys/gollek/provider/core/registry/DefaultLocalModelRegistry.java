package tech.kayys.gollek.provider.core.registry;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelFormatDetector;
import tech.kayys.gollek.spi.registry.LocalModelRegistry;
import tech.kayys.gollek.spi.registry.ModelEntry;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Default CDI implementation of {@link LocalModelRegistry}.
 *
 * <p>Resides in {@code gollek-provider-core} — <strong>not</strong> in
 * {@code gollek-spi} — so it can carry CDI annotations and I/O logic without
 * polluting the pure-contract SPI layer.
 *
 * <h3>Thread-safety</h3>
 * All mutable state is guarded by {@link ConcurrentHashMap} or
 * {@code synchronized} blocks.  {@link #scanAll()} is idempotent and can be
 * called concurrently; it uses {@code computeIfAbsent} to avoid duplicate
 * registration.
 *
 * <h3>Resolution order (see {@link LocalModelRegistry} contract)</h3>
 * <ol>
 *   <li>Exact index hit.</li>
 *   <li>Alias map lookup.</li>
 *   <li>Treat {@code modelRef} as an absolute path.</li>
 *   <li>On-demand scan + exact retry.</li>
 *   <li>Fuzzy filename-stem match across scanned entries.</li>
 * </ol>
 */
@ApplicationScoped
public class DefaultLocalModelRegistry implements LocalModelRegistry {

    private static final Logger LOG = Logger.getLogger(DefaultLocalModelRegistry.class);

    /** Max depth when walking a scan-root directory tree. */
    private static final int SCAN_DEPTH = 3;

    // ── State ─────────────────────────────────────────────────────────────────

    /** Primary index: modelId → entry. */
    private final ConcurrentHashMap<String, ModelEntry> index = new ConcurrentHashMap<>();

    /** Alias map: alias → canonical modelId. */
    private final ConcurrentHashMap<String, String> aliases = new ConcurrentHashMap<>();

    /** Registered scan roots (insertion-ordered, de-duplicated). */
    private final LinkedHashSet<Path> scanRoots = new LinkedHashSet<>();

    // ── LocalModelRegistry — Registration ──────────────────────────────────────────

    @Override
    public synchronized void addScanRoots(Path... paths) {
        for (Path p : paths) {
            if (p == null) continue;
            if (!Files.isDirectory(p)) {
                LOG.debugf("LocalModelRegistry: skipping non-directory scan root — %s", p);
                continue;
            }
            if (scanRoots.add(p)) {
                LOG.infof("LocalModelRegistry: scan root added — %s", p);
            }
        }
    }

    @Override
    public ModelEntry register(String modelId, Path physicalPath, ModelFormat format) {
        Objects.requireNonNull(modelId,      "modelId must not be null");
        Objects.requireNonNull(physicalPath, "physicalPath must not be null");

        ModelFormat resolved = resolveFormat(physicalPath, format);
        long sizeBytes = calculateSize(physicalPath);

        ModelEntry entry = new ModelEntry(
                modelId,
                calculateDisplayName(physicalPath, resolved),
                resolved,
                physicalPath.toAbsolutePath().normalize(),
                sizeBytes,
                resolved.getRuntime(),
                java.util.Map.of(),
                Instant.now());

        index.put(modelId, entry);
        LOG.debugf("LocalModelRegistry: registered '%s' → %s [%s]", modelId, physicalPath, resolved);
        return entry;
    }

    @Override
    public void registerAlias(String alias, String modelId) {
        if (alias == null || alias.isBlank() || modelId == null || modelId.isBlank()) return;
        aliases.put(alias, modelId);
        LOG.debugf("LocalModelRegistry: alias '%s' → '%s'", alias, modelId);
    }

    // ── LocalModelRegistry — Lookup ─────────────────────────────────────────────────

    @Override
    public Optional<ModelEntry> resolve(String modelRef) {
        return resolveAll(modelRef).stream().findFirst();
    }

    @Override
    public List<ModelEntry> resolveAll(String modelRef) {
        if (modelRef == null || modelRef.isBlank()) {
            return List.of();
        }

        List<ModelEntry> results = new ArrayList<>();

        // 1. Exact index hit
        ModelEntry exact = index.get(modelRef);
        if (exact != null) results.add(exact);

        // 2. Alias lookup
        String canonical = aliases.get(modelRef);
        if (canonical != null) {
            ModelEntry aliased = index.get(canonical);
            if (aliased != null) results.add(aliased);
        }

        // 3. Absolute path
        try {
            Path asPath = Path.of(modelRef);
            if (asPath.isAbsolute() && Files.exists(asPath)) {
                results.add(register(modelRef, asPath, null));
            }
        } catch (Exception ignored) { /* not a valid path string */ }

        // 4. On-demand scan
        if (results.isEmpty() && !scanRoots.isEmpty()) {
            scanAll();
            ModelEntry afterScan = index.get(modelRef);
            if (afterScan != null) results.add(afterScan);
        }

        // 5. Short ID match (prefix)
        if (results.isEmpty() && modelRef.length() >= 6 && modelRef.length() <= 8) {
             index.values().stream()
                 .filter(e -> modelRef.equalsIgnoreCase(tech.kayys.gollek.spi.model.ModelUtils.generateShortId(e.modelId())))
                 .findFirst().ifPresent(results::add);
        }

        // 6. Fuzzy matching
        results.addAll(fuzzyMatchAll(modelRef));

        return results.stream().distinct().collect(Collectors.toList());
    }

    private List<ModelEntry> fuzzyMatchAll(String modelRef) {
        String lc = modelRef.toLowerCase(java.util.Locale.ROOT);
        return index.values().stream()
                .filter(e -> {
                    if (e.physicalPath() == null) return false;
                    String pathLower = e.physicalPath().toString().toLowerCase(java.util.Locale.ROOT);
                    String name = e.physicalPath().getFileName().toString().toLowerCase(java.util.Locale.ROOT);
                    String lcNormalized = lc.replace("/", "_").replace("\\", "_");
                    String nameNormalized = name.replace("/", "_").replace("\\", "_");

                    // 1. Exact name match (normalized)
                    if (nameNormalized.equals(lcNormalized)) return true;

                    // 2. Name match without extension (normalized)
                    int dot = nameNormalized.lastIndexOf('.');
                    if (dot > 0 && nameNormalized.substring(0, dot).equals(lcNormalized)) return true;

                    // 3. Path suffix match (for multi-component IDs like "owner/model")
                    if (pathLower.endsWith(lc)) return true;
                    if (pathLower.endsWith(lcNormalized)) return true;
                    
                    return false;
                })
                .collect(Collectors.toList());
    }

    // ── LocalModelRegistry — Query ─────────────────────────────────────────────────

    @Override
    public List<ModelEntry> listAll(ModelFormat format) {
        Stream<ModelEntry> stream = index.values().stream();
        if (format != null) {
            stream = stream.filter(e -> e.format() == format);
        }
        return stream.sorted(Comparator.comparing(ModelEntry::modelId))
                     .collect(Collectors.toUnmodifiableList());
    }

    // ── LocalModelRegistry — Lifecycle ─────────────────────────────────────────────

    @Override
    public void refresh() {
        LOG.debug("LocalModelRegistry: refresh triggered");
        scanAll();
    }

    @Override
    public void clear() {
        index.clear();
        aliases.clear();
        LOG.debug("LocalModelRegistry: cleared");
    }

    // ── Internal — Discovery ──────────────────────────────────────────────────

    private synchronized void scanAll() {
        for (Path root : scanRoots) {
            scanDirectory(root);
        }
    }

    private void scanDirectory(Path dir) {
        if (!Files.isDirectory(dir)) return;
        try (var walk = Files.walk(dir, SCAN_DEPTH)) {
            walk.forEach(this::registerIfRecognised);
        } catch (IOException e) {
            LOG.warnf("LocalModelRegistry: failed to scan %s — %s", dir, e.getMessage());
        }
    }

    private void registerIfRecognised(Path file) {
        ModelFormatDetector.detect(file).ifPresent(fmt -> {
            String key = file.toAbsolutePath().normalize().toString();
                index.computeIfAbsent(key, k -> {
                    LOG.debugf("LocalModelRegistry: discovered [%s] %s", fmt, file);
                    return new ModelEntry(
                            key,
                            calculateDisplayName(file, fmt),
                            fmt,
                            file.toAbsolutePath().normalize(),
                            0L,
                            fmt.getRuntime(),
                            java.util.Map.of(),
                            Instant.now());
                });
        });
    }

    @Override
    public void load(Path path) {
        LOG.debugf("LocalModelRegistry: load from %s (not implemented)", path);
    }

    @Override
    public void save(Path path) {
        LOG.debugf("LocalModelRegistry: save to %s (not implemented)", path);
    }

    private String calculateDisplayName(Path file, ModelFormat format) {
        String filename = file.getFileName().toString();
        if (format == ModelFormat.SAFETENSORS || format == ModelFormat.PYTORCH) {
            // For directory-based models, if the file is a generic marker, use the parent dir name
            if (format.getMarkerFiles().stream().anyMatch(m -> m.equalsIgnoreCase(filename))) {
                Path parent = file.getParent();
                if (parent != null && parent.getFileName() != null) {
                    return parent.getFileName().toString();
                }
            }
        }
        return filename;
    }

    private static ModelFormat resolveFormat(Path path, ModelFormat hint) {
        if (hint != null && hint != ModelFormat.UNKNOWN) {
            return hint;
        }
        return ModelFormatDetector.detect(path).orElse(ModelFormat.UNKNOWN);
    }

    private long calculateSize(Path path) {
        if (path == null || !Files.exists(path)) return 0L;
        try {
            if (Files.isRegularFile(path)) {
                return Files.size(path);
            }
            try (Stream<Path> walk = Files.walk(path)) {
                return walk.filter(Files::isRegularFile)
                        .mapToLong(p -> {
                            try {
                                return Files.size(p);
                            } catch (IOException e) {
                                return 0L;
                            }
                        })
                        .sum();
            }
        } catch (IOException e) {
            return 0L;
        }
    }
}
