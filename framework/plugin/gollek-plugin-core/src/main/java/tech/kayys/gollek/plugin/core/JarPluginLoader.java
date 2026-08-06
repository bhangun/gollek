/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.core;

import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.plugin.descriptor.PluginDescriptor;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.JarFile;
import java.util.stream.Collectors;

import org.jboss.logging.Logger;

/**
 * Loads plugins from JAR files in ~/.gollek/plugins/ directory.
 * 
 * <p>
 * Supports:
 * </p>
 * <ul>
 * <li>Auto-discovery of JAR files in plugin directory</li>
 * <li>Isolated ClassLoader per plugin</li>
 * <li>Hot-reload on file changes</li>
 * <li>Maven dependency resolution (optional)</li>
 * </ul>
 *
 * <h2>Plugin Directory Structure</h2>
 * 
 * <pre>
 * ~/.gollek/plugins/
 * ├── openai-provider.jar          # Cloud provider plugin
 * ├── anthropic-provider.jar       # Cloud provider plugin
 * ├── custom-model.jar             # Custom model plugin
 * └── plugin.json                  # Optional manifest
 * </pre>
 *
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * JarPluginLoader loader = new JarPluginLoader();
 * List<GollekPlugin> plugins = loader.loadFromDirectory(Paths.get("~/.gollek/plugins"));
 * 
 * // Or load specific JAR
 * GollekPlugin plugin = loader.loadFromJar(Paths.get("plugin.jar"));
 * 
 * // Unload plugin
 * loader.unload("plugin-id");
 * }</pre>
 *
 * @since 2.1.0
 */
public class JarPluginLoader {

    private static final Logger LOG = Logger.getLogger(JarPluginLoader.class);

    /**
     * Default plugin directory: ~/.gollek/plugins
     */
    public static final Path DEFAULT_PLUGIN_DIR = Paths.get(
            System.getProperty("user.home"), ".gollek", "plugins");

    /**
     * Loaded plugin ClassLoaders for isolation
     */
    private final Map<String, PluginClassLoaderHandle> loadedClassLoaders = new ConcurrentHashMap<>();

    /**
     * Plugin directory to monitor
     */
    private final Path pluginDirectory;

    /**
     * File watcher for hot-reload
     */
    private WatchService watchService;

    /**
     * Maven dependency resolver
     */
    private final MavenDependencyResolver mavenResolver;

    /**
     * Create loader with default plugin directory.
     */
    public JarPluginLoader() {
        this(DEFAULT_PLUGIN_DIR);
    }

    /**
     * Create loader with custom plugin directory.
     *
     * @param pluginDirectory Directory containing plugin JARs
     */
    public JarPluginLoader(Path pluginDirectory) {
        this.pluginDirectory = pluginDirectory;
        this.mavenResolver = new MavenDependencyResolver();
    }

    /**
     * Create loader with custom Maven resolver.
     *
     * @param pluginDirectory Directory containing plugin JARs
     * @param mavenResolver   Maven dependency resolver
     */
    public JarPluginLoader(Path pluginDirectory, MavenDependencyResolver mavenResolver) {
        this.pluginDirectory = pluginDirectory;
        this.mavenResolver = mavenResolver;
    }

    /**
     * Load all plugins from plugin directory.
     *
     * @return List of loaded plugins
     */
    public List<GollekPlugin> loadAll() {
        if (!Files.exists(pluginDirectory)) {
            LOG.infof("Plugin directory does not exist: %s", pluginDirectory);
            return Collections.emptyList();
        }

        LOG.infof("Loading plugins from: %s", pluginDirectory);

        try {
            List<GollekPlugin> plugins = Files.walk(pluginDirectory, 1)
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".jar"))
                    .map(this::loadFromJar)
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());

            LOG.infof("Loaded %d plugins from JARs", plugins.size());
            return plugins;

        } catch (IOException e) {
            LOG.errorf(e, "Failed to load plugins from %s", pluginDirectory);
            return Collections.emptyList();
        }
    }

    /**
     * Load plugin from specific JAR file.
     *
     * @param jarPath Path to JAR file
     * @return Loaded plugin or null if loading failed
     */
    public GollekPlugin loadFromJar(Path jarPath) {
        if (!Files.exists(jarPath)) {
            LOG.warnf("JAR not found: %s", jarPath);
            return null;
        }

        try {
            LOG.infof("Loading plugin from: %s", jarPath);

            // Load plugin descriptor
            PluginDescriptor descriptor = loadPluginDescriptor(jarPath);

            // Resolve Maven dependencies if specified
            List<File> dependencyJars = Collections.emptyList();
            if (descriptor != null && !descriptor.dependencies().isEmpty()) {
                LOG.infof("Resolving %d Maven dependencies for plugin: %s",
                        descriptor.dependencies().size(), descriptor.id());
                try {
                    dependencyJars = mavenResolver.resolveAll(descriptor.dependencies());
                    LOG.infof("Resolved %d dependency JAR(s)", dependencyJars.size());
                } catch (Exception e) {
                    LOG.errorf(e, "Failed to resolve Maven dependencies for plugin: %s", descriptor.id());
                    // Continue loading without dependencies - may fail at runtime
                }
            }

            // Create ClassLoader with plugin JAR and dependencies
            List<URL> urls = new ArrayList<>();
            urls.add(jarPath.toUri().toURL());
            for (File depJar : dependencyJars) {
                urls.add(depJar.toURI().toURL());
            }

            PluginClassLoaderHandle handle = new PluginClassLoaderHandle(urls.toArray(new URL[0]));

            // Load main plugin class
            Class<?> pluginClass = handle.classLoader.loadClass(descriptor.mainClass());

            // Instantiate plugin
            GollekPlugin plugin = instantiatePlugin(pluginClass);

            // Store ClassLoader for later unloading
            loadedClassLoaders.put(plugin.id(), handle);

            LOG.infof("Successfully loaded plugin: %s (version %s, dependencies: %d)",
                    plugin.id(), plugin.version(), dependencyJars.size());
            return plugin;

        } catch (Exception e) {
            LOG.errorf(e, "Failed to load plugin from JAR: %s", jarPath);
            return null;
        }
    }

    /**
     * Unload plugin by ID.
     *
     * @param pluginId Plugin ID to unload
     * @return true if unloaded successfully
     */
    public boolean unload(String pluginId) {
        PluginClassLoaderHandle handle = loadedClassLoaders.remove(pluginId);
        if (handle == null) {
            LOG.warnf("Plugin not found for unloading: %s", pluginId);
            return false;
        }

        try {
            handle.classLoader.close();
            LOG.infof("Unloaded plugin: %s", pluginId);
            return true;
        } catch (IOException e) {
            LOG.errorf(e, "Failed to unload plugin: %s", pluginId);
            return false;
        }
    }

    /**
     * Start watching for plugin changes (hot-reload).
     */
    public void startWatching() {
        if (!Files.exists(pluginDirectory)) {
            try {
                Files.createDirectories(pluginDirectory);
                LOG.infof("Created plugin directory: %s", pluginDirectory);
            } catch (IOException e) {
                LOG.errorf(e, "Failed to create plugin directory");
                return;
            }
        }

        try {
            watchService = FileSystems.getDefault().newWatchService();
            pluginDirectory.register(watchService,
                    StandardWatchEventKinds.ENTRY_CREATE,
                    StandardWatchEventKinds.ENTRY_DELETE,
                    StandardWatchEventKinds.ENTRY_MODIFY);

            LOG.infof("Started watching plugin directory: %s", pluginDirectory);

            // Start watch thread
            Thread watchThread = new Thread(this::watchLoop, "Plugin-Watch-Thread");
            watchThread.setDaemon(true);
            watchThread.start();

        } catch (IOException e) {
            LOG.errorf(e, "Failed to start file watcher");
        }
    }

    /**
     * Stop watching for changes.
     */
    public void stopWatching() {
        if (watchService != null) {
            try {
                watchService.close();
                LOG.info("Stopped watching plugin directory");
            } catch (IOException e) {
                LOG.errorf(e, "Failed to close watch service");
            }
        }
    }

    /**
     * Get all loaded plugin IDs.
     *
     * @return Set of plugin IDs
     */
    public Set<String> getLoadedPluginIds() {
        return Collections.unmodifiableSet(loadedClassLoaders.keySet());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Load plugin descriptor from JAR.
     *
     * @param jarPath Path to JAR file
     * @return Plugin descriptor or null if not found
     */
    private PluginDescriptor loadPluginDescriptor(Path jarPath) {
        try (JarFile jarFile = new JarFile(jarPath.toFile())) {
            var entry = jarFile.getEntry("plugin.json");
            if (entry != null) {
                // Read and parse plugin.json
                String json = new String(jarFile.getInputStream(entry).readAllBytes());
                return PluginDescriptor.fromJson(json);
            }
        } catch (IOException e) {
            LOG.warnf(e, "Failed to load plugin.json from: %s", jarPath);
        }

        // Return default descriptor
        return new PluginDescriptor(
                "unknown-plugin",
                "Unknown Plugin",
                "1.0.0",
                "Plugin without descriptor",
                "Unknown",
                "tech.kayys.gollek.plugin.UnknownPlugin",
                Collections.emptyList(),
                Collections.emptyList(),
                Collections.emptyMap());
    }

    private void watchLoop() {
        try {
            WatchKey key = watchService.take();

            for (WatchEvent<?> event : key.pollEvents()) {
                WatchEvent.Kind<?> kind = event.kind();
                Path changedFile = (Path) event.context();

                if (changedFile.toString().endsWith(".jar")) {
                    LOG.infof("Plugin file changed: %s", changedFile);

                    if (kind == StandardWatchEventKinds.ENTRY_CREATE ||
                            kind == StandardWatchEventKinds.ENTRY_MODIFY) {
                        // Reload plugin
                        Path fullPath = pluginDirectory.resolve(changedFile);
                        GollekPlugin plugin = loadFromJar(fullPath);
                        if (plugin != null) {
                            LOG.infof("Hot-reloaded plugin: %s", plugin.id());
                        }
                    } else if (kind == StandardWatchEventKinds.ENTRY_DELETE) {
                        // Unload plugin
                        String pluginId = changedFile.getFileName().toString()
                                .replace(".jar", "");
                        unload(pluginId);
                    }
                }
            }

            key.reset();

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private PluginMetadata loadMetadata(ClassLoader classLoader, Path jarPath) throws IOException {
        // Try to load plugin.json from JAR
        try (JarFile jarFile = new JarFile(jarPath.toFile())) {
            var entry = jarFile.getEntry("plugin.json");
            if (entry != null) {
                // Load and parse plugin.json
                // For now, use defaults
                return new PluginMetadata(
                        "unknown",
                        "Unknown Plugin",
                        "1.0.0",
                        "Plugin without descriptor",
                        "Unknown",
                        "",
                        "MIT",
                        new String[0]);
            }
        }

        // Default metadata
        return new PluginMetadata(
                "unknown",
                "Unknown Plugin",
                "1.0.0",
                "Plugin without descriptor",
                "Unknown",
                "",
                "MIT",
                new String[0]);
    }

    @SuppressWarnings("unchecked")
    private GollekPlugin instantiatePlugin(Class<?> pluginClass) throws Exception {
        // Try to find no-arg constructor
        Constructor<?> constructor = pluginClass.getDeclaredConstructor();
        constructor.setAccessible(true);
        return (GollekPlugin) constructor.newInstance();
    }

    /**
     * ClassLoader handle for tracking loaded plugins.
     */
    private static class PluginClassLoaderHandle {
        final URLClassLoader classLoader;
        final long loadTime;

        PluginClassLoaderHandle(URL... urls) {
            this.classLoader = new URLClassLoader(urls,
                    ClassLoader.getSystemClassLoader().getParent());
            this.loadTime = System.currentTimeMillis();
        }
    }
}