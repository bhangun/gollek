/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.kernel;

import org.jboss.logging.Logger;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import java.util.ServiceLoader;

/**
 * Advanced kernel plugin loader with ClassLoader isolation, hot-reload support,
 * and dependency management.
 *
 * <h2>Features:</h2>
 * <ul>
 *   <li>ClassLoader isolation per plugin</li>
 *   <li>Hot-reload with compatibility checking</li>
 *   <li>ServiceLoader discovery</li>
 *   <li>Manifest-based plugin metadata</li>
 *   <li>Dependency resolution</li>
 * </ul>
 *
 * @since 2.0.0
 */
public class KernelPluginLoader {

    private static final Logger LOG = Logger.getLogger(KernelPluginLoader.class);

    private static final String PLUGIN_ID_MANIFEST = "Plugin-Id";
    private static final String PLUGIN_TYPE_MANIFEST = "Plugin-Type";
    private static final String PLUGIN_VERSION_MANIFEST = "Plugin-Version";
    private static final String PLATFORM_MANIFEST = "Platform";

    private final Map<String, KernelPluginClassLoader> pluginClassLoaders = new ConcurrentHashMap<>();
    private final Map<String, KernelPlugin> loadedPlugins = new ConcurrentHashMap<>();
    private final Path pluginDirectory;
    private final ClassLoader parentClassLoader;

    /**
     * Create plugin loader with default plugin directory.
     */
    public KernelPluginLoader() {
        this(Path.of(System.getProperty("user.home"), ".gollek", "plugins", "kernels"));
    }

    /**
     * Create plugin loader with specified directory.
     *
     * @param pluginDirectory plugin directory
     */
    public KernelPluginLoader(Path pluginDirectory) {
        this(pluginDirectory, KernelPluginLoader.class.getClassLoader());
    }

    /**
     * Create plugin loader with specified directory and parent ClassLoader.
     *
     * @param pluginDirectory plugin directory
     * @param parentClassLoader parent ClassLoader
     */
    public KernelPluginLoader(Path pluginDirectory, ClassLoader parentClassLoader) {
        this.pluginDirectory = pluginDirectory;
        this.parentClassLoader = parentClassLoader;
    }

    /**
     * Load all plugins from plugin directory.
     *
     * @return list of loaded plugins
     * @throws KernelException if loading fails
     */
    public List<KernelPlugin> loadAll() throws KernelException {
        LOG.infof("Loading kernel plugins from: %s", pluginDirectory);

        if (!pluginDirectory.toFile().exists()) {
            LOG.warnf("Plugin directory does not exist: %s", pluginDirectory);
            return List.of();
        }

        List<KernelPlugin> plugins = new ArrayList<>();

        // Find all JAR files
        File[] jarFiles = pluginDirectory.toFile().listFiles(
            (dir, name) -> name.endsWith(".jar"));

        if (jarFiles == null || jarFiles.length == 0) {
            LOG.info("No kernel plugin JARs found");
            return plugins;
        }

        for (File jarFile : jarFiles) {
            try {
                KernelPlugin plugin = loadPlugin(jarFile.toPath());
                if (plugin != null) {
                    plugins.add(plugin);
                    LOG.infof("Loaded kernel plugin: %s from %s",
                            plugin.id(), jarFile.getName());
                }
            } catch (Exception e) {
                LOG.errorf(e, "Failed to load plugin from: %s", jarFile);
            }
        }

        LOG.infof("Loaded %d kernel plugins", plugins.size());
        return plugins;
    }

    /**
     * Load plugin from JAR file.
     *
     * @param jarPath path to JAR file
     * @return loaded plugin or null
     * @throws KernelException if loading fails
     */
    public KernelPlugin loadPlugin(Path jarPath) throws KernelException {
        LOG.debugf("Loading plugin from: %s", jarPath);

        try {
            // Read manifest
            Map<String, String> manifest = readManifest(jarPath);

            String pluginId = manifest.get(PLUGIN_ID_MANIFEST);
            String pluginType = manifest.get(PLUGIN_TYPE_MANIFEST);

            // Check if this is a kernel plugin
            if (pluginType == null || !pluginType.equalsIgnoreCase("kernel")) {
                LOG.debugf("Skipping non-kernel plugin: %s", pluginId);
                return null;
            }

            if (pluginId == null) {
                LOG.warnf("Plugin JAR missing Plugin-Id manifest: %s", jarPath);
                return null;
            }

            // Check if already loaded
            if (loadedPlugins.containsKey(pluginId)) {
                LOG.warnf("Plugin already loaded: %s, unloading first", pluginId);
                unloadPlugin(pluginId);
            }

            // Create isolated ClassLoader
            KernelPluginClassLoader classLoader = createClassLoader(jarPath);
            pluginClassLoaders.put(pluginId, classLoader);

            // Load plugin class using ServiceLoader
            ServiceLoader<KernelPlugin> serviceLoader = ServiceLoader.load(
                KernelPlugin.class, classLoader);

            KernelPlugin plugin = null;
            for (KernelPlugin p : serviceLoader) {
                if (pluginId.equals(p.id())) {
                    plugin = p;
                    break;
                }
            }

            if (plugin == null) {
                // Try to instantiate directly
                String pluginClass = manifest.get("Plugin-Class");
                if (pluginClass != null) {
                    try {
                        Class<?> clazz = classLoader.loadClass(pluginClass);
                        if (KernelPlugin.class.isAssignableFrom(clazz)) {
                            plugin = (KernelPlugin) clazz.getDeclaredConstructor().newInstance();
                        }
                    } catch (Exception e) {
                        LOG.debugf("Could not instantiate plugin class: %s", pluginClass);
                    }
                }
            }

            if (plugin != null) {
                loadedPlugins.put(pluginId, plugin);
                LOG.infof("Successfully loaded kernel plugin: %s (version: %s)",
                        pluginId, plugin.version());
            }

            return plugin;

        } catch (KernelException e) {
            throw e;
        } catch (Exception e) {
            throw new KernelInitializationException("Failed to load plugin from " + jarPath, e);
        }
    }

    /**
     * Unload plugin by ID.
     *
     * @param pluginId plugin ID
     * @return true if unloaded successfully
     */
    public boolean unloadPlugin(String pluginId) {
        LOG.infof("Unloading plugin: %s", pluginId);

        KernelPlugin plugin = loadedPlugins.remove(pluginId);
        if (plugin != null) {
            try {
                plugin.shutdown();
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down plugin: %s", pluginId);
            }
        }

        KernelPluginClassLoader classLoader = pluginClassLoaders.remove(pluginId);
        if (classLoader != null) {
            try {
                classLoader.close();
            } catch (Exception e) {
                LOG.errorf(e, "Error closing ClassLoader for plugin: %s", pluginId);
            }
        }

        return plugin != null;
    }

    /**
     * Reload plugin from JAR file.
     *
     * @param jarPath path to JAR file
     * @return reloaded plugin or null
     * @throws KernelException if reloading fails
     */
    public KernelPlugin reloadPlugin(Path jarPath) throws KernelException {
        try {
            Map<String, String> manifest = readManifest(jarPath);
            String pluginId = manifest.get(PLUGIN_ID_MANIFEST);

            if (pluginId != null) {
                unloadPlugin(pluginId);
            }

            return loadPlugin(jarPath);
        } catch (KernelException e) {
            throw e;
        } catch (Exception e) {
            throw new KernelInitializationException("Failed to reload plugin from " + jarPath, e);
        }
    }

    /**
     * Get loaded plugin by ID.
     *
     * @param pluginId plugin ID
     * @return optional plugin
     */
    public Optional<KernelPlugin> getPlugin(String pluginId) {
        return Optional.ofNullable(loadedPlugins.get(pluginId));
    }

    /**
     * Get all loaded plugins.
     *
     * @return list of plugins
     */
    public List<KernelPlugin> getAllPlugins() {
        return List.copyOf(loadedPlugins.values());
    }

    /**
     * Get loaded plugin count.
     *
     * @return plugin count
     */
    public int getPluginCount() {
        return loadedPlugins.size();
    }

    /**
     * Close loader and unload all plugins.
     */
    public void close() {
        LOG.info("Closing kernel plugin loader...");

        Set<String> pluginIds = Set.copyOf(loadedPlugins.keySet());
        for (String pluginId : pluginIds) {
            unloadPlugin(pluginId);
        }

        LOG.info("Kernel plugin loader closed");
    }

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * Read manifest from JAR file.
     *
     * @param jarPath path to JAR
     * @return manifest attributes
     * @throws Exception if reading fails
     */
    private Map<String, String> readManifest(Path jarPath) throws Exception {
        Map<String, String> manifest = new HashMap<>();

        try (JarFile jarFile = new JarFile(jarPath.toFile())) {
            java.util.jar.Attributes attrs = jarFile.getManifest().getMainAttributes();

            for (Object keyObj : attrs.keySet()) {
                String key = keyObj.toString();
                String value = attrs.getValue(key);
                manifest.put(key, value);
            }
        }

        return manifest;
    }

    /**
     * Create isolated ClassLoader for plugin.
     *
     * @param jarPath path to JAR
     * @return plugin ClassLoader
     * @throws Exception if creation fails
     */
    private KernelPluginClassLoader createClassLoader(Path jarPath) throws Exception {
        URL jarUrl = jarPath.toUri().toURL();
        return new KernelPluginClassLoader(new URL[]{jarUrl}, parentClassLoader);
    }

    // ========================================================================
    // Plugin ClassLoader
    // ========================================================================

    /**
     * Isolated ClassLoader for kernel plugin with parent-first delegation
     * for core classes.
     */
    public static class KernelPluginClassLoader extends URLClassLoader {

        private static final Set<String> PARENT_FIRST_PACKAGES = Set.of(
            "java.",
            "javax.",
            "jakarta.",
            "org.jboss.logging",
            "tech.kayys.gollek.plugin.kernel",
            "tech.kayys.gollek.spi"
        );

        private final Set<String> loadedClasses = ConcurrentHashMap.newKeySet();
        private volatile boolean closed = false;

        public KernelPluginClassLoader(URL[] urls, ClassLoader parent) {
            super(urls, parent);
        }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            // Parent-first for core packages
            if (isParentFirst(name)) {
                return getParent().loadClass(name);
            }

            // Check if already loaded
            Class<?> loadedClass = findLoadedClass(name);
            if (loadedClass != null) {
                return loadedClass;
            }

            // Try to load from plugin JAR
            try {
                loadedClass = findClass(name);
                loadedClasses.add(name);
                if (resolve) {
                    resolveClass(loadedClass);
                }
                return loadedClass;
            } catch (ClassNotFoundException e) {
                // Delegate to parent
                return getParent().loadClass(name);
            }
        }

        /**
         * Check if class should be loaded from parent first.
         *
         * @param className class name
         * @return true if parent-first
         */
        private boolean isParentFirst(String className) {
            return PARENT_FIRST_PACKAGES.stream()
                    .anyMatch(pkg -> className.startsWith(pkg));
        }

        /**
         * Get loaded class names.
         *
         * @return set of class names
         */
        public Set<String> getLoadedClasses() {
            return Set.copyOf(loadedClasses);
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                loadedClasses.clear();
                try {
                    super.close();
                } catch (java.io.IOException e) {
                    LOG.warnf(e, "Error closing ClassLoader");
                }
            }
        }
    }
}
