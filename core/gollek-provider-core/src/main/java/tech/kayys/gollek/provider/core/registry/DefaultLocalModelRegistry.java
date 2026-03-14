package tech.kayys.gollek.provider.core.registry;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelFormatDetector;
import tech.kayys.gollek.spi.registry.LocalModelRegistry;

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
        ModelEntry entry = new ModelEntry(
                modelId,
                physicalPath.toAbsolutePath().normalize(),
                resolved,
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
        if (modelRef == null || modelRef.isBlank()) {
            return Optional.empty();
        }

        // 1. Exact index hit
        ModelEntry exact = index.get(modelRef);
        if (exact != null) return Optional.of(exact);

        // 2. Alias lookup
        String canonical = aliases.get(modelRef);
        if (canonical != null) {
            ModelEntry aliased = index.get(canonical);
            if (aliased != null) return Optional.of(aliased);
        }

        // 3. Absolute path
        try {
            Path asPath = Path.of(modelRef);
            if (asPath.isAbsolute() && Files.isRegularFile(asPath)) {
                return Optional.of(register(modelRef, asPath, null));
            }
        } catch (Exception ignored) { /* not a valid path string */ }

        // 4. On-demand scan then exact retry
        if (!scanRoots.isEmpty()) {
            scanAll();
            ModelEntry afterScan = index.get(modelRef);
            if (afterScan != null) return Optional.of(afterScan);
        }

        // 5. Fuzzy filename-stem match
        return fuzzyMatch(modelRef);
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
            walk.filter(Files::isRegularFile)
                .forEach(this::registerIfRecognised);
        } catch (IOException e) {
            LOG.warnf("LocalModelRegistry: failed to scan %s — %s", dir, e.getMessage());
        }
    }

    private void registerIfRecognised(Path file) {
        ModelFormatDetector.detect(file).ifPresent(fmt -> {
            String key = file.toAbsolutePath().normalize().toString();
            index.computeIfAbsent(key, k -> {
                LOG.debugf("LocalModelRegistry: discovered [%s] %s", fmt, file);
                return new ModelEntry(key, file.toAbsolutePath().normalize(), fmt, Instant.now());
            });
        });
    }

    // ── Internal — Utilities ──────────────────────────────────────────────────

    private static ModelFormat resolveFormat(Path path, ModelFormat hint) {
        if (hint != null && hint != ModelFormat.UNKNOWN) {
            return hint;
        }
        return ModelFormatDetector.detect(path).orElse(ModelFormat.UNKNOWN);
    }

    private Optional<ModelEntry> fuzzyMatch(String modelRef) {
        String lc = modelRef.toLowerCase(java.util.Locale.ROOT);
        return index.values().stream()
                .filter(e -> {
                    if (e.physicalPath() == null) return false;
                    String name = e.physicalPath().getFileName().toString()
                                   .toLowerCase(java.util.Locale.ROOT);
                    if (name.equals(lc)) return true;
                    int dot = name.lastIndexOf('.');
                    return dot > 0 && name.substring(0, dot).equals(lc);
                })
                .findFirst();
    }
}
