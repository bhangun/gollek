/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.sdk.route;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

/**
 * Small TSV-backed cache for local route artifact equivalence lookups.
 */
public final class RouteArtifactCache {
    private RouteArtifactCache() {
    }

    public static Optional<Path> find(
            String cacheDirProperty,
            String cacheEnabledProperty,
            String fileName,
            String cacheKey,
            Predicate<Path> validator) {
        if (!enabled(cacheEnabledProperty) || blank(cacheKey) || validator == null) {
            return Optional.empty();
        }
        Path cacheFile = cacheFile(cacheDirProperty, fileName);
        if (!Files.isRegularFile(cacheFile)) {
            return Optional.empty();
        }
        try {
            for (String line : Files.readAllLines(cacheFile, StandardCharsets.UTF_8)) {
                String[] parts = line.split("\t", 2);
                if (parts.length != 2 || !cacheKey.equals(parts[0])) {
                    continue;
                }
                Path cached = Path.of(parts[1]).toAbsolutePath().normalize();
                if (validator.test(cached)) {
                    return Optional.of(cached);
                }
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
        return Optional.empty();
    }

    public static void put(
            String cacheDirProperty,
            String cacheEnabledProperty,
            String fileName,
            String cacheKey,
            Path selected,
            Predicate<Path> validator) {
        if (!enabled(cacheEnabledProperty)
                || blank(cacheKey)
                || selected == null
                || validator == null
                || !validator.test(selected)) {
            return;
        }
        Path cacheFile = cacheFile(cacheDirProperty, fileName);
        String entry = cacheKey + "\t" + selected.toAbsolutePath().normalize();
        try {
            Files.createDirectories(cacheFile.getParent());
            List<String> lines = Files.isRegularFile(cacheFile)
                    ? Files.readAllLines(cacheFile, StandardCharsets.UTF_8)
                    : List.of();
            List<String> updated = new ArrayList<>(lines.size() + 1);
            for (String line : lines) {
                if (!line.startsWith(cacheKey + "\t")) {
                    updated.add(line);
                }
            }
            updated.add(entry);
            Files.write(cacheFile, updated, StandardCharsets.UTF_8);
        } catch (Exception ignored) {
            // A stale/missing cache should never block a valid runtime route.
        }
    }

    private static boolean enabled(String propertyName) {
        String configured = System.getProperty(propertyName);
        return configured == null || configured.isBlank() || Boolean.parseBoolean(configured);
    }

    private static Path cacheFile(String cacheDirProperty, String fileName) {
        String configured = System.getProperty(cacheDirProperty);
        Path dir = configured == null || configured.isBlank()
                ? Path.of(System.getProperty("user.home"), ".gollek", "cache")
                : Path.of(configured.trim());
        return dir.resolve(fileName);
    }

    private static boolean blank(String value) {
        return value == null || value.isBlank();
    }
}
