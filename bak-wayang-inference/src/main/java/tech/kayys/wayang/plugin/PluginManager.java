package tech.kayys.golek.plugin;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.jar.JarFile;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.golek.engine.LlamaEngine;

public class PluginManager {
    private static final Logger log = LoggerFactory.getLogger(PluginManager.class);
    
    private final Map<String, Plugin> plugins = new ConcurrentHashMap<>();
    private final Map<String, PluginMetadata> metadata = new ConcurrentHashMap<>();
    private final Map<String, List<PluginEventListener>> eventListeners = new ConcurrentHashMap<>();
    private final Map<String, Object> sharedData = new ConcurrentHashMap<>();
    private final Path pluginsDirectory;
    private final Path dataDirectory;
    private final LlamaEngine engine;
    
    public PluginManager(LlamaEngine engine, Path pluginsDirectory, Path dataDirectory) {
        this.engine = engine;
        this.pluginsDirectory = pluginsDirectory;
        this.dataDirectory = dataDirectory;
    }
    
    /**
     * Discover and load all plugins from the plugins directory
     */
    public void loadPlugins() throws PluginException {
        log.info("Loading plugins from: {}", pluginsDirectory);
        
        try {
            if (!Files.exists(pluginsDirectory)) {
                Files.createDirectories(pluginsDirectory);
                log.info("Created plugins directory");
                return;
            }
            
            try (Stream<Path> paths = Files.list(pluginsDirectory)) {
                paths.filter(path -> path.toString().endsWith(".jar"))
                     .forEach(this::loadPluginFromJar);
            }
            
            // Sort plugins by dependencies
            List<GolekPlugin> sortedPlugins = sortByDependencies();
            
            // Initialize and start plugins
            for (Plugin plugin : sortedPlugins) {
                try {
                    PluginContext context = createContext(plugin);
                    plugin.initialize(context);
                    plugin.start();
                    log.info("Plugin started: {} v{}", plugin.getName(), plugin.getVersion());
                } catch (Exception e) {
                    log.error("Failed to start plugin: {}", plugin.getId(), e);
                    plugins.remove(plugin.getId());
                }
            }
            
            log.info("Loaded {} plugins", plugins.size());
            
        } catch (Exception e) {
            throw new PluginException("Failed to load plugins", e);
        }
    }
    
    private void loadPluginFromJar(Path jarPath) {
        try {
            URLClassLoader classLoader = new URLClassLoader(
                new URL[]{jarPath.toUri().toURL()},
                getClass().getClassLoader()
            );
            
            try (JarFile jarFile = new JarFile(jarPath.toFile())) {
                // Look for plugin.properties
                var entry = jarFile.getEntry("plugin.properties");
                if (entry == null) {
                    log.warn("No plugin.properties in {}", jarPath.getFileName());
                    return;
                }
                
                Properties props = new Properties();
                props.load(jarFile.getInputStream(entry));
                
                String mainClass = props.getProperty("main.class");
                if (mainClass == null) {
                    log.warn("No main.class in plugin.properties: {}", jarPath.getFileName());
                    return;
                }
                
                Class<?> pluginClass = classLoader.loadClass(mainClass);
                Plugin plugin = (Plugin) pluginClass.getDeclaredConstructor().newInstance();
                
                registerPlugin(plugin, new PluginMetadata(
                    jarPath,
                    props,
                    classLoader
                ));
                
                log.info("Discovered plugin: {} v{}", plugin.getName(), plugin.getVersion());
            }
            
        } catch (Exception e) {
            log.error("Failed to load plugin from {}", jarPath, e);
        }
    }
    
    public void registerPlugin(Plugin plugin, PluginMetadata meta) {
        plugins.put(plugin.getId(), plugin);
        metadata.put(plugin.getId(), meta);
    }
    
    public void unloadPlugin(String pluginId) throws PluginException {
        Plugin plugin = plugins.get(pluginId);
        if (plugin == null) {
            throw new PluginException("Plugin not found: " + pluginId);
        }
        
        try {
            plugin.stop();
            plugins.remove(pluginId);
            metadata.remove(pluginId);
            log.info("Plugin unloaded: {}", plugin.getName());
        } catch (Exception e) {
            throw new PluginException("Failed to unload plugin: " + pluginId, e);
        }
    }
    
    public void stopAllPlugins() {
        plugins.values().forEach(plugin -> {
            try {
                plugin.stop();
            } catch (Exception e) {
                log.error("Error stopping plugin: {}", plugin.getId(), e);
            }
        });
        plugins.clear();
    }
    
    private List<GolekPlugin> sortByDependencies() {
        List<GolekPlugin> sorted = new ArrayList<>();
        Set<String> processed = new HashSet<>();
        
        for (Plugin plugin : plugins.values()) {
            addWithDependencies(plugin, sorted, processed);
        }
        
        return sorted;
    }
    
    private void addWithDependencies(Plugin plugin, List<GolekPlugin> sorted, Set<String> processed) {
        if (processed.contains(plugin.getId())) {
            return;
        }
        
        for (String depId : plugin.getDependencies()) {
            Plugin dep = plugins.get(depId);
            if (dep != null) {
                addWithDependencies(dep, sorted, processed);
            }
        }
        
        sorted.add(plugin);
        processed.add(plugin.getId());
    }
    
    private PluginContext createContext(Plugin plugin) {
        return new PluginContext() {
            @Override
            public LlamaEngine getEngine() {
                return engine;
            }
            
            @Override
            public Map<String, Object> getConfig() {
                PluginMetadata meta = metadata.get(plugin.getId());
                return meta != null ? meta.getConfigMap() : Map.of();
            }
            
            @Override
            public <T> T getConfigValue(String key, Class<T> type) {
                Object value = getConfig().get(key);
                return type.cast(value);
            }
            
            @Override
            public String getDataDirectory() {
                Path pluginDataDir = dataDirectory.resolve(plugin.getId());
                try {
                    Files.createDirectories(pluginDataDir);
                } catch (Exception e) {
                    log.error("Failed to create data directory for plugin", e);
                }
                return pluginDataDir.toString();
            }
            
            @Override
            public Object getSharedData(String key) {
                return sharedData.get(key);
            }
            
            @Override
            public void setSharedData(String key, Object value) {
                sharedData.put(key, value);
            }
            
            @Override
            public <T extends Plugin> T getPlugin(String pluginId, Class<T> type) {
                Plugin p = plugins.get(pluginId);
                return type.isInstance(p) ? type.cast(p) : null;
            }
            
            @Override
            public void addEventListener(String eventType, PluginEventListener listener) {
                eventListeners.computeIfAbsent(eventType, k -> new CopyOnWriteArrayList<>())
                             .add(listener);
            }
            
            @Override
            public void emitEvent(String eventType, Object data) {
                PluginEvent event = new PluginEvent(eventType, data, plugin.getId());
                List<PluginEventListener> listeners = eventListeners.get(eventType);
                if (listeners != null) {
                    listeners.forEach(listener -> {
                        try {
                            listener.onEvent(event);
                        } catch (Exception e) {
                            log.error("Error in event listener", e);
                        }
                    });
                }
            }
        };
    }
    
    public <T extends Plugin> List<T> getPluginsByType(Class<T> type) {
        return plugins.values().stream()
            .filter(type::isInstance)
            .map(type::cast)
            .toList();
    }
    
    public Plugin getPlugin(String pluginId) {
        return plugins.get(pluginId);
    }
    
    public Map<String, Plugin> getAllPlugins() {
        return Collections.unmodifiableMap(plugins);
    }
    
    public void emitGlobalEvent(String eventType, Object data) {
        PluginEvent event = new PluginEvent(eventType, data, "system");
        List<PluginEventListener> listeners = eventListeners.get(eventType);
        if (listeners != null) {
            listeners.forEach(listener -> {
                try {
                    listener.onEvent(event);
                } catch (Exception e) {
                    log.error("Error in event listener", e);
                }
            });
        }
    }
    
    private record PluginMetadata(
        Path jarPath,
        Properties properties,
        ClassLoader classLoader
    ) {
        Map<String, Object> getConfigMap() {
            Map<String, Object> config = new HashMap<>();
            properties.forEach((k, v) -> config.put(k.toString(), v));
            return config;
        }
    }
}
