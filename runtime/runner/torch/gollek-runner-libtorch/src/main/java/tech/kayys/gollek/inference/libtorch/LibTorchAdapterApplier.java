package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.inference.libtorch.util.SafetensorsLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

@ApplicationScoped
public class LibTorchAdapterApplier {

    private static final Logger log = Logger.getLogger(LibTorchAdapterApplier.class);

    @Inject
    SafetensorsLoader safetensorsLoader;

    @Inject
    LibTorchMetrics metrics;

    private final ConcurrentMap<Path, CachedPairIndex> pairIndexCache = new ConcurrentHashMap<>();

    public int applyRuntimeLora(TorchScriptRunner runner, Path adapterPath, float scale) {
        if (runner == null) {
            throw new IllegalArgumentException("runner cannot be null");
        }
        if (adapterPath == null) {
            throw new IllegalArgumentException("adapterPath cannot be null");
        }

        LibTorchBinding binding = LibTorchBinding.getInstance();
        MethodHandle applyFn = binding.bind(
                LibTorchBinding.JIT_APPLY_LORA,
                LibTorchBinding.JIT_APPLY_LORA_DESC);

        Path normalizedPath = adapterPath.toAbsolutePath().normalize();
        Map<String, LoraPairKeys> pairs = resolvePairIndex(normalizedPath);
        if (pairs.isEmpty()) {
            throw new IllegalArgumentException(
                    "No LoRA A/B tensor pairs found in adapter file: " + normalizedPath);
        }

        int applied = 0;
        for (var entry : pairs.entrySet()) {
            String baseName = entry.getKey();
            LoraPairKeys pair = entry.getValue();
            try (TorchTensor loraA = safetensorsLoader.loadTensor(normalizedPath, pair.aKey());
                    TorchTensor loraB = safetensorsLoader.loadTensor(normalizedPath, pair.bKey());
                    Arena arena = Arena.ofConfined()) {
                int rc = tryApplyWithFallbackNames(
                        applyFn, runner.moduleHandle(), baseName, loraA, loraB, scale, arena);
                if (rc != 0) {
                    throw new IllegalArgumentException(
                            "Failed applying LoRA for base '" + baseName + "' (code=" + rc + ")");
                }
                applied++;
            } catch (RuntimeException e) {
                throw e;
            } catch (Throwable t) {
                throw new RuntimeException("Native LoRA apply failed for base '" + baseName + "'", t);
            }
        }

        log.infof("Applied %d LoRA updates from %s (scale=%.4f)",
                applied, normalizedPath.getFileName(), scale);
        return applied;
    }

    static Map<String, LoraPairKeys> discoverPairNames(Set<String> tensorNames) {
        Map<String, LoraPairKeys> pairs = new LinkedHashMap<>();
        for (String key : tensorNames) {
            String base = extractLoraABase(key);
            if (base == null) {
                continue;
            }

            String bWithWeight = base + ".lora_B.weight";
            String bPlain = base + ".lora_B";
            String bKey = tensorNames.contains(bWithWeight) ? bWithWeight
                    : (tensorNames.contains(bPlain) ? bPlain : null);
            if (bKey == null) {
                continue;
            }
            pairs.put(base, new LoraPairKeys(key, bKey));
        }
        return pairs;
    }

    private int tryApplyWithFallbackNames(MethodHandle applyFn,
            MemorySegment moduleHandle,
            String baseName,
            TorchTensor loraA,
            TorchTensor loraB,
            float scale,
            Arena arena) throws Throwable {
        int lastCode = 2;
        for (String candidate : candidateBaseNames(baseName)) {
            MemorySegment baseNameStr = arena.allocateFrom(candidate);
            int rc = (int) applyFn.invoke(
                    moduleHandle,
                    baseNameStr,
                    loraA.nativeHandle(),
                    loraB.nativeHandle(),
                    scale);
            if (rc == 0) {
                return 0;
            }
            // Retry only on "parameter not found"; fail fast on other native errors.
            if (rc != 2) {
                return rc;
            }
            lastCode = rc;
        }
        return lastCode;
    }

    private List<String> candidateBaseNames(String baseName) {
        LinkedHashSet<String> names = new LinkedHashSet<>();
        names.add(baseName);
        names.add(stripPrefix(baseName, "base_model.model."));
        names.add(stripPrefix(baseName, "model."));
        names.add(stripPrefix(baseName, "base_model."));
        names.removeIf(s -> s == null || s.isBlank());
        return new ArrayList<>(names);
    }

    private static String stripPrefix(String value, String prefix) {
        if (value == null || prefix == null) {
            return value;
        }
        return value.startsWith(prefix) ? value.substring(prefix.length()) : value;
    }

    private static String extractLoraABase(String key) {
        if (key == null) {
            return null;
        }
        if (key.endsWith(".lora_A.weight")) {
            return key.substring(0, key.length() - ".lora_A.weight".length());
        }
        if (key.endsWith(".lora_A")) {
            return key.substring(0, key.length() - ".lora_A".length());
        }
        return null;
    }

    private Map<String, LoraPairKeys> resolvePairIndex(Path adapterPath) {
        try {
            long size = Files.size(adapterPath);
            long modifiedAt = Files.getLastModifiedTime(adapterPath).toMillis();
            CachedPairIndex cached = pairIndexCache.get(adapterPath);
            if (cached != null && cached.size() == size && cached.modifiedAtMillis() == modifiedAt) {
                if (metrics != null) {
                    metrics.recordAdapterPairCacheHit();
                }
                return cached.pairs();
            }

            Map<String, LoraPairKeys> computed = Collections.unmodifiableMap(
                    discoverPairNames(safetensorsLoader.inspect(adapterPath).keySet()));
            pairIndexCache.put(adapterPath, new CachedPairIndex(size, modifiedAt, computed));
            if (metrics != null) {
                metrics.recordAdapterPairCacheMiss();
            }
            return computed;
        } catch (Exception e) {
            throw new RuntimeException("Failed to inspect adapter tensor metadata: " + adapterPath, e);
        }
    }

    record CachedPairIndex(long size, long modifiedAtMillis, Map<String, LoraPairKeys> pairs) {
    }

    record LoraPairKeys(String aKey, String bKey) {
    }
}
