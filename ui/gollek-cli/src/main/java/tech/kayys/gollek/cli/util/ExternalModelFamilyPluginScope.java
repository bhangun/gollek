/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Scoped bridge for execution paths that still read model-family metadata from
 * the process-global registry.
 */
public final class ExternalModelFamilyPluginScope implements AutoCloseable {
    private final ExternalPluginClasspathScope classpathScope;
    private final List<String> detachablePluginIds;
    private final Map<String, ModelFamilyPlugin> shadowedPlugins;

    private ExternalModelFamilyPluginScope(
            ExternalPluginClasspathScope classpathScope,
            List<String> detachablePluginIds,
            Map<String, ModelFamilyPlugin> shadowedPlugins) {
        this.classpathScope = classpathScope == null ? ExternalPluginClasspathScope.empty() : classpathScope;
        this.detachablePluginIds = detachablePluginIds == null ? List.of() : List.copyOf(detachablePluginIds);
        this.shadowedPlugins = shadowedPlugins == null ? Map.of() : Map.copyOf(shadowedPlugins);
    }

    public static ExternalModelFamilyPluginScope empty() {
        return new ExternalModelFamilyPluginScope(ExternalPluginClasspathScope.empty(), List.of(), Map.of());
    }

    public static ExternalModelFamilyPluginScope attach(
            List<String> classpathEntries,
            Class<?> parentAnchor,
            PluginAvailabilityChecker pluginChecker) {
        return attach(classpathEntries, List.of(), parentAnchor, pluginChecker);
    }

    public static ExternalModelFamilyPluginScope attach(
            List<String> classpathEntries,
            List<String> pluginDirectoryEntries,
            Class<?> parentAnchor,
            PluginAvailabilityChecker pluginChecker) {
        return attachScope(
                ExternalPluginClasspathScope.open(classpathEntries, pluginDirectoryEntries, parentAnchor),
                pluginChecker);
    }

    public static ExternalModelFamilyPluginScope attachParsed(
            List<Path> classpath,
            Class<?> parentAnchor,
            PluginAvailabilityChecker pluginChecker) {
        return attachScope(
                ExternalPluginClasspathScope.openParsed(classpath, parentAnchor),
                pluginChecker);
    }

    private static ExternalModelFamilyPluginScope attachScope(
            ExternalPluginClasspathScope classpathScope,
            PluginAvailabilityChecker pluginChecker) {
        if (classpathScope == null || !classpathScope.active()) {
            return new ExternalModelFamilyPluginScope(classpathScope, List.of(), Map.of());
        }

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        Map<String, ModelFamilyPlugin> before = modelFamilyPluginSnapshot(registry);
        try {
            if (pluginChecker != null) {
                pluginChecker.getModelFamilyPluginRegistry(null);
            } else {
                java.util.ServiceLoader.load(ModelFamilyPlugin.class).forEach(registry::register);
            }
            before = modelFamilyPluginSnapshot(registry);
            java.util.ServiceLoader.load(ModelFamilyPlugin.class, classpathScope.discoveryClassLoader()).forEach(registry::register);
            Map<String, ModelFamilyPlugin> after = modelFamilyPluginSnapshot(registry);

            List<String> detachablePluginIds = new ArrayList<>();
            Map<String, ModelFamilyPlugin> shadowedPlugins = new LinkedHashMap<>();
            for (Map.Entry<String, ModelFamilyPlugin> entry : after.entrySet()) {
                ModelFamilyPlugin previous = before.get(entry.getKey());
                if (previous == null) {
                    detachablePluginIds.add(entry.getKey());
                } else if (previous != entry.getValue()) {
                    detachablePluginIds.add(entry.getKey());
                    shadowedPlugins.put(entry.getKey(), previous);
                }
            }
            return new ExternalModelFamilyPluginScope(classpathScope, detachablePluginIds, shadowedPlugins);
        } catch (RuntimeException | Error error) {
            restoreGlobalRegistry(registry, before);
            closeQuietly(classpathScope);
            throw error;
        }
    }

    public boolean active() {
        return classpathScope.active();
    }

    public List<Path> classpath() {
        return classpathScope.classpath();
    }

    public List<String> displayClasspath() {
        return classpathScope.displayClasspath();
    }

    public List<String> detachablePluginIds() {
        return detachablePluginIds;
    }

    @Override
    public void close() {
        try {
            if (active()) {
                ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
                for (String pluginId : detachablePluginIds) {
                    registry.unregister(pluginId);
                }
                for (ModelFamilyPlugin plugin : shadowedPlugins.values()) {
                    registry.register(plugin);
                }
            }
        } finally {
            closeQuietly(classpathScope);
        }
    }

    private static void restoreGlobalRegistry(
            ModelFamilyPluginRegistry registry,
            Map<String, ModelFamilyPlugin> before) {
        Map<String, ModelFamilyPlugin> after = modelFamilyPluginSnapshot(registry);
        for (String pluginId : after.keySet()) {
            if (!before.containsKey(pluginId)) {
                registry.unregister(pluginId);
            }
        }
        before.values().forEach(registry::register);
    }

    private static void closeQuietly(ExternalPluginClasspathScope scope) {
        try {
            if (scope != null) {
                scope.close();
            }
        } catch (Exception ignored) {
            // A close failure should not turn a successful inference into a CLI failure.
        }
    }

    private static Map<String, ModelFamilyPlugin> modelFamilyPluginSnapshot(ModelFamilyPluginRegistry registry) {
        Map<String, ModelFamilyPlugin> plugins = new LinkedHashMap<>();
        if (registry == null) {
            return plugins;
        }
        for (ModelFamilyPlugin plugin : registry.all()) {
            String key = modelFamilyPluginKey(plugin);
            if (!key.isBlank()) {
                plugins.putIfAbsent(key, plugin);
            }
        }
        return plugins;
    }

    private static String modelFamilyPluginKey(ModelFamilyPlugin plugin) {
        if (plugin == null) {
            return "";
        }
        try {
            String id = plugin.id();
            if (id != null && !id.isBlank()) {
                return normalizeModelFamilyPluginId(id);
            }
        } catch (RuntimeException ignored) {
        }
        return normalizeModelFamilyPluginId(plugin.getClass().getName() + "@" + System.identityHashCode(plugin));
    }

    private static String normalizeModelFamilyPluginId(String value) {
        String normalized = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        if (normalized.isBlank() || normalized.startsWith("model-family/")) {
            return normalized;
        }
        return "model-family/" + normalized;
    }
}
