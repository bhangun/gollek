/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.runtime.ModelRuntimeTraitsResolver;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.gollek.safetensor.utils.SafetensorWriter;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

final class MetalF16WeightCache {
    private static final Logger LOG = Logger.getLogger(MetalF16WeightCache.class);
    private static final String DISABLE_METAL_F16_DISK_CACHE_PROPERTY = "gollek.safetensor.disable_metal_f16_disk_cache";
    private static final String ENABLE_GEMMA4_METAL_F16_DISK_CACHE_PROPERTY = "gollek.safetensor.enable_gemma4_metal_f16_disk_cache";
    private static final String METAL_F16_DISK_CACHE_MAX_BYTES_PROPERTY = "gollek.safetensor.metal_f16_disk_cache_max_bytes";
    private static final String METAL_F16_DISK_CACHE_MIN_FREE_BYTES_PROPERTY = "gollek.safetensor.metal_f16_disk_cache_min_free_bytes";
    private static final long GIB = 1024L * 1024L * 1024L;
    private static final long DEFAULT_METAL_F16_DISK_CACHE_MAX_BYTES = 12L * GIB;
    private static final long DEFAULT_METAL_F16_DISK_CACHE_MIN_FREE_BYTES = 2L * GIB;
    private static final long METAL_F16_DISK_CACHE_WRITE_SAFETY_BYTES = 256L * 1024L * 1024L;

    private final SafetensorLoaderFacade safetensorLoader;
    private final AccelWeightBridge weightBridge;

    MetalF16WeightCache(SafetensorLoaderFacade safetensorLoader, AccelWeightBridge weightBridge) {
        this.safetensorLoader = safetensorLoader;
        this.weightBridge = weightBridge;
    }

    Map<String, AccelTensor> maybeUse(Path modelPath, Map<String, AccelTensor> weights, ModelConfig config,
            ModelRuntimeTraits traits) {
        if (Boolean.getBoolean(DISABLE_METAL_F16_DISK_CACHE_PROPERTY) || !isNativeMetalRuntimeActive()) {
            return weights;
        }
        ModelRuntimeTraits effectiveTraits = ModelRuntimeTraitsResolver.resolve(config, traits);
        if (effectiveTraits.gemma4Text()
                && !Boolean.getBoolean(ENABLE_GEMMA4_METAL_F16_DISK_CACHE_PROPERTY)) {
            LOG.debugf("Skipping Metal F16 weight cache for Gemma-4 text; BF16->F16 parity is experimental.");
            return weights;
        }
        long cacheMaxBytes = Long.getLong(
                METAL_F16_DISK_CACHE_MAX_BYTES_PROPERTY,
                DEFAULT_METAL_F16_DISK_CACHE_MAX_BYTES);
        if (cacheMaxBytes <= 0L) {
            return weights;
        }
        Path cachePath = cachePath(modelPath);
        long cacheBudgetBytes = Files.isRegularFile(cachePath)
                ? cacheMaxBytes
                : availableCacheBudget(cachePath, cacheMaxBytes);
        Map<String, AccelTensor> cacheableWeights = selectCacheWeights(weights, cacheBudgetBytes);
        if (cacheableWeights.isEmpty()) {
            return weights;
        }
        long cacheBytes = estimateCacheBytes(cacheableWeights);
        if (cacheBytes <= 0L || cacheBytes > cacheMaxBytes) {
            return weights;
        }

        Map<String, AccelTensor> cached = tryLoad(cachePath, weights, cacheableWeights.keySet());
        if (cached != null) {
            return cached;
        }
        if (!hasEnoughCacheSpace(cachePath, cacheBytes)) {
            return weights;
        }

        Path tempPath = tempCachePath(cachePath);
        try {
            Files.createDirectories(cachePath.getParent());
            Map<String, AccelTensor> converted = new LinkedHashMap<>(cacheableWeights.size() * 2);
            try {
                for (Map.Entry<String, AccelTensor> entry : cacheableWeights.entrySet()) {
                    AccelTensor tensor = entry.getValue();
                    AccelTensor cacheTensor = tensor.quantType() == AccelTensor.QuantType.BF16
                            ? tensor.toF16CachedUpTo(cacheMaxBytes)
                            : tensor;
                    if (cacheTensor == null) {
                        return weights;
                    }
                    converted.put(entry.getKey(), cacheTensor);
                }

                Files.deleteIfExists(tempPath);
                SafetensorWriter.save(tempPath, converted);
                moveReplacing(tempPath, cachePath);
                LOG.infof("DirectInferenceEngine: saved Metal F16 weight cache [%s]", cachePath);
                cached = tryLoad(cachePath, weights, cacheableWeights.keySet());
                if (cached != null) {
                    return cached;
                }
            } finally {
                // Converted tensors are cached on the original BF16 tensors, so the normal
                // model unload path owns them.
            }
        } catch (Exception e) {
            try {
                Files.deleteIfExists(tempPath);
            } catch (IOException cleanupError) {
                e.addSuppressed(cleanupError);
            }
            LOG.warnf(e, "Failed to persist Metal F16 weight cache for %s", modelPath);
        }
        return weights;
    }

    private Map<String, AccelTensor> tryLoad(
            Path cachePath,
            Map<String, AccelTensor> expected,
            Set<String> expectedCachedNames) {
        if (!Files.isRegularFile(cachePath)) {
            return null;
        }
        try (SafetensorShardSession cacheSession = safetensorLoader.open(cachePath)) {
            Map<String, AccelTensor> cached = weightBridge.bridgeAll(cacheSession);
            if (!cached.keySet().containsAll(expectedCachedNames)) {
                cached.values().forEach(AccelTensor::close);
                LOG.warnf("Ignoring stale Metal F16 weight cache with missing tensors: %s", cachePath);
                return null;
            }

            Map<String, AccelTensor> merged = new LinkedHashMap<>(expected);
            Set<String> usedCachedNames = new HashSet<>();
            for (String name : expectedCachedNames) {
                AccelTensor source = expected.get(name);
                AccelTensor replacement = cached.get(name);
                if (!isValidCacheReplacement(source, replacement)) {
                    cached.values().forEach(AccelTensor::close);
                    LOG.warnf("Ignoring stale Metal F16 weight cache with mismatched tensor '%s': %s", name, cachePath);
                    return null;
                }
                AccelTensor old = merged.put(name, replacement);
                if (old != null) {
                    old.close();
                }
                usedCachedNames.add(name);
            }

            cached.entrySet().stream()
                    .filter(entry -> !usedCachedNames.contains(entry.getKey()))
                    .map(Map.Entry::getValue)
                    .forEach(AccelTensor::close);
            LOG.infof("DirectInferenceEngine: using Metal F16 weight cache [%s] (%d tensors)",
                    cachePath, usedCachedNames.size());
            return merged;
        } catch (Exception e) {
            LOG.warnf(e, "Failed to load Metal F16 weight cache from %s", cachePath);
        }
        return null;
    }

    private boolean isValidCacheReplacement(AccelTensor source, AccelTensor cached) {
        if (source == null || cached == null || !Arrays.equals(source.shape(), cached.shape())) {
            return false;
        }
        if (source.quantType() != AccelTensor.QuantType.BF16
                && source.quantType() != AccelTensor.QuantType.F16) {
            return false;
        }
        return cached.quantType() == AccelTensor.QuantType.F16;
    }

    private Map<String, AccelTensor> selectCacheWeights(Map<String, AccelTensor> weights, long cacheBudgetBytes) {
        Map<String, AccelTensor> selected = new LinkedHashMap<>();
        long usedBytes = 0L;
        List<Map.Entry<String, AccelTensor>> candidates = weights.entrySet().stream()
                .filter(entry -> isCacheCandidate(entry.getKey(), entry.getValue()))
                .sorted(Comparator.comparingInt(entry -> cachePriority(entry.getKey())))
                .toList();
        for (Map.Entry<String, AccelTensor> entry : candidates) {
            String name = entry.getKey();
            AccelTensor tensor = entry.getValue();
            long bytes = cacheByteSize(tensor);
            if (cacheBudgetBytes > 0L && usedBytes + bytes > cacheBudgetBytes) {
                continue;
            }
            selected.put(name, tensor);
            usedBytes += bytes;
        }
        return selected;
    }

    private boolean isCacheCandidate(String name, AccelTensor tensor) {
        if (tensor == null || tensor.rank() < 2) {
            return false;
        }
        if (tensor.quantType() != AccelTensor.QuantType.BF16
                && tensor.quantType() != AccelTensor.QuantType.F16) {
            return false;
        }
        if (name.startsWith("model.audio_tower.")
                || name.startsWith("model.vision_tower.")
                || name.startsWith("model.embed_audio.")
                || name.startsWith("model.embed_vision.")) {
            return false;
        }
        return !name.endsWith("embed_tokens_per_layer.weight");
    }

    private int cachePriority(String name) {
        if (name.endsWith("embed_tokens.weight")) {
            return 0;
        }
        if (name.contains(".mlp.") || name.contains(".feed_forward.")) {
            return 1;
        }
        if (name.contains(".self_attn.") || name.contains(".attention.")) {
            return 2;
        }
        if (name.startsWith("model.language_model.layers.") || name.startsWith("model.layers.")) {
            return 3;
        }
        return 4;
    }

    private long availableCacheBudget(Path cachePath, long cacheMaxBytes) {
        Path cacheDir = cachePath.getParent();
        if (cacheDir == null) {
            return cacheMaxBytes;
        }
        long minFreeBytes = Long.getLong(
                METAL_F16_DISK_CACHE_MIN_FREE_BYTES_PROPERTY,
                DEFAULT_METAL_F16_DISK_CACHE_MIN_FREE_BYTES);
        if (minFreeBytes < 0L) {
            minFreeBytes = 0L;
        }
        try {
            Files.createDirectories(cacheDir);
            long usableBytes = Files.getFileStore(cacheDir).getUsableSpace();
            long reclaimableBytes = staleTempCacheBytes(cachePath);
            long effectiveUsableBytes = saturatingAdd(usableBytes, reclaimableBytes);
            long reserveBytes = saturatingAdd(minFreeBytes, METAL_F16_DISK_CACHE_WRITE_SAFETY_BYTES);
            long diskBudgetBytes = Math.max(0L, effectiveUsableBytes - reserveBytes);
            return Math.min(cacheMaxBytes, diskBudgetBytes);
        } catch (IOException e) {
            LOG.warnf(e, "Could not determine Metal F16 cache budget for %s", cacheDir);
            return cacheMaxBytes;
        }
    }

    private boolean hasEnoughCacheSpace(Path cachePath, long cacheBytes) {
        Path cacheDir = cachePath.getParent();
        if (cacheDir == null) {
            return true;
        }
        long minFreeBytes = Long.getLong(
                METAL_F16_DISK_CACHE_MIN_FREE_BYTES_PROPERTY,
                DEFAULT_METAL_F16_DISK_CACHE_MIN_FREE_BYTES);
        if (minFreeBytes < 0L) {
            minFreeBytes = 0L;
        }
        try {
            Files.createDirectories(cacheDir);
            long usableBytes = Files.getFileStore(cacheDir).getUsableSpace();
            long reclaimableBytes = staleTempCacheBytes(cachePath);
            long effectiveUsableBytes = saturatingAdd(usableBytes, reclaimableBytes);
            long reserveBytes = saturatingAdd(minFreeBytes, METAL_F16_DISK_CACHE_WRITE_SAFETY_BYTES);
            long requiredBytes = Math.addExact(cacheBytes, reserveBytes);
            if (effectiveUsableBytes >= requiredBytes) {
                return true;
            }
            LOG.warnf(
                    "Skipping Metal F16 weight cache because free space is too low. Need ~%s for cache plus %s reserve, have %s%s.",
                    formatGib(cacheBytes),
                    formatGib(reserveBytes),
                    formatGib(usableBytes),
                    reclaimableBytes > 0L ? " (+" + formatGib(reclaimableBytes) + " reclaimable temp)" : "");
        } catch (ArithmeticException e) {
            LOG.warnf("Skipping Metal F16 weight cache because required space overflowed for %s", cachePath);
        } catch (IOException e) {
            LOG.warnf(e, "Skipping Metal F16 weight cache because cache directory space could not be checked: %s",
                    cacheDir);
        }
        return false;
    }

    private Path tempCachePath(Path cachePath) {
        return cachePath.resolveSibling(cachePath.getFileName() + ".tmp");
    }

    private long staleTempCacheBytes(Path cachePath) {
        Path tempPath = tempCachePath(cachePath);
        try {
            return Files.isRegularFile(tempPath) ? Files.size(tempPath) : 0L;
        } catch (IOException e) {
            return 0L;
        }
    }

    private long saturatingAdd(long left, long right) {
        try {
            return Math.addExact(left, right);
        } catch (ArithmeticException e) {
            return Long.MAX_VALUE;
        }
    }

    private String formatGib(long bytes) {
        return String.format(Locale.ROOT, "%.1f GiB", bytes / (double) GIB);
    }

    private void moveReplacing(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private long estimateCacheBytes(Map<String, AccelTensor> weights) {
        long total = 0L;
        for (AccelTensor tensor : weights.values()) {
            if (tensor == null) {
                return Long.MAX_VALUE;
            }
            long bytes = cacheByteSize(tensor);
            try {
                total = Math.addExact(total, bytes);
            } catch (ArithmeticException e) {
                return Long.MAX_VALUE;
            }
        }
        return total;
    }

    private long cacheByteSize(AccelTensor tensor) {
        return switch (tensor.quantType()) {
            case BF16, F16 -> tensor.halfStorageByteSize();
            case F32 -> tensor.dequantizedByteSize();
            case INT8, FP8 -> tensor.numel();
            case INT4, NF4 -> (tensor.numel() + 1L) / 2L;
        };
    }

    private Path cachePath(Path modelPath) {
        String gollekHome = System.getProperty("gollek.home",
                Path.of(System.getProperty("user.home"), ".gollek").toString());
        String key = sha256Hex(modelCacheFingerprint(modelPath));
        return Path.of(gollekHome, "cache", "metal-f16", key + ".safetensors");
    }

    private String modelCacheFingerprint(Path modelPath) {
        Path resolved = modelPath.toAbsolutePath().normalize();
        StringBuilder fingerprint = new StringBuilder(resolved.toString());
        try {
            if (Files.isRegularFile(resolved)) {
                appendFileFingerprint(fingerprint, resolved);
            } else if (Files.isDirectory(resolved)) {
                try (var stream = Files.walk(resolved, 2)) {
                    List<Path> safetensors = stream
                            .filter(Files::isRegularFile)
                            .filter(this::isSafetensorFile)
                            .sorted()
                            .toList();
                    for (Path path : safetensors) {
                        appendFileFingerprint(fingerprint, path);
                    }
                }
            }
        } catch (IOException e) {
            LOG.warnf(e, "Falling back to path-only Metal F16 cache key for %s", resolved);
        }
        return fingerprint.toString();
    }

    private boolean isSafetensorFile(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".safetensors") || name.endsWith(".safetensor");
    }

    private void appendFileFingerprint(StringBuilder fingerprint, Path path) throws IOException {
        fingerprint.append('|')
                .append(path.toAbsolutePath().normalize())
                .append(':')
                .append(Files.size(path))
                .append(':')
                .append(Files.getLastModifiedTime(path).toMillis());
    }

    private String sha256Hex(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder out = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                out.append(Character.forDigit((b >>> 4) & 0xF, 16));
                out.append(Character.forDigit(b & 0xF, 16));
            }
            return out.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private boolean isNativeMetalRuntimeActive() {
        try {
            MetalBinding.initialize();
            MetalBinding binding = MetalBinding.getInstance();
            return binding != null && binding.isRuntimeActive();
        } catch (Throwable ignored) {
            return false;
        }
    }
}
