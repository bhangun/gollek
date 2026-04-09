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

package tech.kayys.gollek.sdk.core;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.sdk.exception.NoPluginsAvailableException;
import tech.kayys.gollek.sdk.exception.PluginNotAvailableException;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.stream.Collectors;

/**
 * SDK-level plugin availability checker with manifest-based capability discovery.
 *
 * <p>This class provides centralized plugin availability checking for both standalone and microservice
 * deployments. It uses manifest-based metadata discovery instead of hardcoded mappings, making the
 * system flexible, agnostic, and extensible.</p>
 *
 * <h2>Deployment Modes</h2>
 * <ul>
 *   <li><b>STANDALONE</b> - All plugins built-in, compiled in one JAR</li>
 *   <li><b>MICROSERVICE</b> - Dynamic plugin loading from plugin directory</li>
 *   <li><b>HYBRID</b> - Built-in + dynamic plugins</li>
 * </ul>
 *
 * <h2>Manifest-Based Discovery</h2>
 * <p>Plugins declare capabilities in JAR manifest:</p>
 * <pre>
 * Plugin-Id: gguf-runner
 * Plugin-Type: runner
 * Plugin-Capabilities: gguf-inference, llama-architecture
 * Plugin-Deployment: standalone,microservice
 * Plugin-GPU-Requirement: CUDA 11.0+
 * Plugin-Performance-Speedup: 2-3x
 * </pre>
 *
 * @since 2.0.0
 */
public class PluginAvailabilityChecker {

    private static final Logger LOG = Logger.getLogger(PluginAvailabilityChecker.class);

    private static final String PLUGIN_DIRECTORY_PROPERTY = "gollek.plugin.directory";
    private static final String DEFAULT_PLUGIN_DIRECTORY = System.getProperty("user.home") + "/.gollek/plugins";
    private static final String DEPLOYMENT_MODE_PROPERTY = "gollek.deployment.mode";

    private final ProviderRegistry providerRegistry;
    private final String pluginDirectory;
    private final DeploymentMode deploymentMode;
    private final Map<String, PluginDescriptor> discoveredPlugins = new ConcurrentHashMap<>();
    private final Map<String, List<PluginDescriptor>> capabilityIndex = new ConcurrentHashMap<>();
    private final Map<String, Boolean> capabilityCache = new ConcurrentHashMap<>();

    /**
     * Create plugin availability checker.
     *
     * @param providerRegistry provider registry instance
     */
    public PluginAvailabilityChecker(ProviderRegistry providerRegistry) {
        this.providerRegistry = providerRegistry;
        this.pluginDirectory = System.getProperty(PLUGIN_DIRECTORY_PROPERTY, DEFAULT_PLUGIN_DIRECTORY);
        this.deploymentMode = detectDeploymentMode();
        
        // Discover and index plugins
        discoverPlugins();
        buildCapabilityIndex();
        
        LOG.infof("Plugin Availability Checker initialized");
        LOG.infof("  Plugin directory: %s", pluginDirectory);
        LOG.infof("  Deployment mode: %s", deploymentMode);
        LOG.infof("  Discovered plugins: %d", discoveredPlugins.size());
        LOG.infof("  Indexed capabilities: %d", capabilityIndex.size());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Plugin Discovery
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Discover all available plugins.
     */
    private void discoverPlugins() {
        // Discover via ServiceLoader
        ServiceLoader<GollekPlugin> loader = ServiceLoader.load(GollekPlugin.class);
        for (GollekPlugin plugin : loader) {
            try {
                PluginDescriptor descriptor = loadPluginDescriptor(plugin.getClass());
                if (descriptor != null) {
                    discoveredPlugins.put(plugin.id(), descriptor);
                    LOG.debugf("Discovered plugin: %s (type=%s, capabilities=%s)", 
                        plugin.id(), descriptor.getType(), descriptor.getCapabilities());
                }
            } catch (Exception e) {
                LOG.debugf("Failed to load descriptor for plugin %s: %s", plugin.id(), e.getMessage());
            }
        }

        // Discover from plugin directory (for MICROSERVICE and HYBRID modes)
        if (deploymentMode != DeploymentMode.STANDALONE) {
            discoverFromDirectory();
        }
    }

    /**
     * Discover plugins from plugin directory.
     */
    private void discoverFromDirectory() {
        Path pluginDir = Paths.get(pluginDirectory);
        
        if (!Files.exists(pluginDir)) {
            return;
        }

        try {
            Files.list(pluginDir)
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".jar"))
                .forEach(jarPath -> {
                    try {
                        PluginDescriptor descriptor = loadDescriptorFromJar(jarPath);
                        if (descriptor != null) {
                            discoveredPlugins.put(descriptor.getId(), descriptor);
                            LOG.debugf("Discovered plugin from JAR: %s", descriptor.getId());
                        }
                    } catch (Exception e) {
                        LOG.debugf("Failed to load plugin from %s: %s", jarPath, e.getMessage());
                    }
                });
        } catch (IOException e) {
            LOG.debugf("Error listing plugin directory: %s", e.getMessage());
        }
    }

    /**
     * Load plugin descriptor from plugin class.
     */
    private PluginDescriptor loadPluginDescriptor(Class<?> pluginClass) {
        try {
            URL resource = pluginClass.getProtectionDomain().getCodeSource().getLocation();
            if (resource != null && resource.getProtocol().equals("file")) {
                Path jarPath = Paths.get(resource.toURI());
                return loadDescriptorFromJar(jarPath);
            }
        } catch (Exception e) {
            LOG.debugf("Failed to load descriptor for %s: %s", pluginClass.getName(), e.getMessage());
        }
        
        // Fallback: create minimal descriptor from annotations/metadata
        return createMinimalDescriptor(pluginClass);
    }

    /**
     * Load descriptor from JAR file.
     */
    private PluginDescriptor loadDescriptorFromJar(Path jarPath) throws IOException {
        try (JarFile jarFile = new JarFile(jarPath.toFile())) {
            Manifest manifest = jarFile.getManifest();
            if (manifest != null) {
                return new PluginDescriptor(manifest);
            }
        }
        return null;
    }

    /**
     * Create minimal descriptor when manifest not available.
     */
    private PluginDescriptor createMinimalDescriptor(Class<?> pluginClass) {
        String pluginId = pluginClass.getSimpleName().toLowerCase();
        String pluginType = inferPluginType(pluginClass);
        
        return PluginDescriptor.builder()
            .id(pluginId)
            .type(pluginType)
            .name(pluginClass.getSimpleName())
            .providerClass(pluginClass.getName())
            .capability(pluginId) // Default capability = plugin ID
            .supportedDeployments(DeploymentMode.HYBRID)
            .build();
    }

    /**
     * Infer plugin type from class name/package.
     */
    private String inferPluginType(Class<?> pluginClass) {
        String packageName = pluginClass.getPackage().getName().toLowerCase();
        String className = pluginClass.getSimpleName().toLowerCase();
        
        if (packageName.contains("kernel") || className.contains("kernel")) {
            return "kernel";
        } else if (packageName.contains("runner") || className.contains("runner")) {
            return "runner";
        } else if (packageName.contains("provider") || className.contains("provider")) {
            return "provider";
        } else if (packageName.contains("optimization") || className.contains("optimization")) {
            return "optimization";
        } else if (packageName.contains("feature") || className.contains("feature")) {
            return "feature";
        }
        
        return "plugin"; // Default
    }

    /**
     * Build capability index for fast lookup.
     */
    private void buildCapabilityIndex() {
        for (PluginDescriptor plugin : discoveredPlugins.values()) {
            for (String capability : plugin.getCapabilities()) {
                capabilityIndex
                    .computeIfAbsent(capability, k -> new ArrayList<>())
                    .add(plugin);
            }
        }
        
        LOG.infof("Built capability index with %d capabilities", capabilityIndex.size());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Capability Checking
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Check if a specific capability is available.
     *
     * @param capability capability name (discovered from plugin manifests)
     * @return true if capability is available
     */
    public boolean hasCapability(String capability) {
        return capabilityCache.computeIfAbsent(capability, this::checkCapability);
    }

    private boolean checkCapability(String capability) {
        List<PluginDescriptor> plugins = capabilityIndex.get(capability.toLowerCase());
        
        if (plugins == null || plugins.isEmpty()) {
            return false;
        }
        
        // Check if at least one plugin supporting this capability is available
        for (PluginDescriptor plugin : plugins) {
            if (isPluginAvailable(plugin)) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * Check if plugin is available.
     */
    private boolean isPluginAvailable(PluginDescriptor plugin) {
        // Check deployment mode compatibility
        if (!plugin.getSupportedDeployments().contains(deploymentMode)) {
            return false;
        }
        
        // Check if plugin is loaded
        if (providerRegistry != null) {
            try {
                return providerRegistry.getProvider(plugin.getId()).isPresent();
            } catch (Exception e) {
                LOG.debugf("Error checking plugin %s: %s", plugin.getId(), e.getMessage());
            }
        }
        
        // Check if plugin class is loadable
        try {
            if (plugin.getProviderClass() != null) {
                Class.forName(plugin.getProviderClass());
                return true;
            }
        } catch (ClassNotFoundException e) {
            // Plugin not loadable
        }
        
        return false;
    }

    /**
     * Get plugins that provide a capability.
     *
     * @param capability capability name
     * @return list of plugin descriptors
     */
    public List<PluginDescriptor> getPluginsForCapability(String capability) {
        List<PluginDescriptor> plugins = capabilityIndex.get(capability.toLowerCase());
        
        if (plugins == null) {
            return Collections.emptyList();
        }
        
        return plugins.stream()
            .filter(this::isPluginAvailable)
            .collect(Collectors.toList());
    }

    /**
     * Get required plugins for a capability (from manifest dependencies).
     *
     * @param capability capability name
     * @return formatted list of required plugins
     */
    public String getRequiredPluginsForCapability(String capability) {
        List<PluginDescriptor> plugins = getPluginsForCapability(capability);
        
        if (plugins.isEmpty()) {
            // Try to find plugin that declares this capability
            for (PluginDescriptor plugin : discoveredPlugins.values()) {
                if (plugin.hasCapability(capability)) {
                    StringBuilder required = new StringBuilder();
                    required.append("   • ").append(plugin.getId());
                    
                    // Add dependencies
                    if (!plugin.getDependencies().isEmpty()) {
                        for (String dep : plugin.getDependencies()) {
                            required.append("\n   • ").append(dep);
                        }
                    }
                    
                    // Add GPU requirements from metadata
                    if (plugin.getMetadata().containsKey("gpu_requirement")) {
                        required.append("\n     Requires: ").append(plugin.getMetadata("gpu_requirement"));
                    }
                    if (plugin.getMetadata().containsKey("min_compute_capability")) {
                        required.append("\n     Minimum compute capability: ").append(plugin.getMetadata("min_compute_capability"));
                    }
                    
                    return required.toString();
                }
            }
            return "";
        }
        
        StringBuilder required = new StringBuilder();
        for (PluginDescriptor plugin : plugins) {
            required.append("   • ").append(plugin.getId());
            
            // Add GPU requirements from metadata
            if (plugin.getMetadata().containsKey("gpu_requirement")) {
                required.append(" (").append(plugin.getMetadata("gpu_requirement")).append(")");
            }
            
            required.append("\n");
        }
        
        return required.toString();
    }

    /**
     * List all available capabilities.
     *
     * @return set of capability names
     */
    public Set<String> getAvailableCapabilities() {
        Set<String> capabilities = new HashSet<>();
        
        for (PluginDescriptor plugin : discoveredPlugins.values()) {
            if (isPluginAvailable(plugin)) {
                capabilities.addAll(plugin.getCapabilities());
            }
        }
        
        return capabilities;
    }

    /**
     * List all discovered plugins.
     *
     * @return collection of plugin descriptors
     */
    public Collection<PluginDescriptor> getAllPlugins() {
        return Collections.unmodifiableCollection(discoveredPlugins.values());
    }

    /**
     * Get plugin by ID.
     *
     * @param pluginId plugin ID
     * @return optional plugin descriptor
     */
    public Optional<PluginDescriptor> getPlugin(String pluginId) {
        return Optional.ofNullable(discoveredPlugins.get(pluginId));
    }

    // ───────────────────────────────────────────────────────────────────────
    // Provider Checking (delegates to ProviderRegistry)
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Check if any providers are available.
     */
    public boolean hasProviders() {
        try {
            if (providerRegistry == null) {
                return false;
            }
            var providers = providerRegistry.getAllProviders();
            return providers != null && !providers.isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking providers: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Check if a specific provider is available.
     */
    public boolean hasProvider(String providerId) {
        try {
            if (providerRegistry == null) {
                return false;
            }
            return providerRegistry.getProvider(providerId).isPresent();
        } catch (Exception e) {
            LOG.debugf("Error checking provider %s: %s", providerId, e.getMessage());
            return false;
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Error Messages
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Get comprehensive error message when no plugins are available.
     */
    public String getNoPluginsError() {
        StringBuilder message = new StringBuilder();
        message.append("╔═══════════════════════════════════════════════════════════╗\n");
        message.append("║  ⚠️  NO PLUGINS AVAILABLE                                ║\n");
        message.append("╚═══════════════════════════════════════════════════════════╝\n\n");
        message.append("The Gollek platform uses a modular deployment strategy.\n\n");

        if (deploymentMode == DeploymentMode.STANDALONE) {
            message.append("📦 STANDALONE MODE: No plugins were included in this build.\n\n");
            message.append("   To add plugins:\n");
            message.append("   1. Rebuild with plugins: mvn clean install -Pinclude-plugins\n");
            message.append("   2. Or switch to microservice mode\n\n");
        } else {
            message.append("📦 INSTALLATION OPTIONS:\n\n");
            message.append("   Option 1: Download Plugins\n");
            message.append("   ────────────────────────────\n");
            message.append("   1. Visit: https://gollek.ai/plugins\n");
            message.append("   2. Download plugins you need\n");
            message.append("   3. Place in: ").append(pluginDirectory).append("\n");
            message.append("   4. Restart application\n\n");

            message.append("   Option 2: Use Package Manager\n");
            message.append("   ──────────────────────────────\n");
            message.append("   gollek install --all\n");
            message.append("   gollek install <plugin-id>\n\n");
        }

        message.append("📁 PLUGIN DIRECTORY: ").append(pluginDirectory).append("\n");
        message.append("   Status: ").append(Files.exists(Paths.get(pluginDirectory)) ? "✓ Exists" : "✗ Does not exist").append("\n\n");

        // List available capabilities
        Set<String> capabilities = getAvailableCapabilities();
        if (!capabilities.isEmpty()) {
            message.append("🔍 AVAILABLE CAPABILITIES:\n");
            for (String cap : capabilities) {
                message.append("   • ").append(cap).append("\n");
            }
            message.append("\n");
        }

        message.append("💡 TIP: Plugins declare capabilities in manifest.\n");
        message.append("   Check plugin JAR manifest for Plugin-Capabilities entry.\n");

        return message.toString();
    }

    /**
     * Get error message for missing provider.
     */
    public String getProviderNotFoundError(String providerId) {
        StringBuilder message = new StringBuilder();
        message.append("❌ Provider '").append(providerId).append("' is not available.\n\n");
        message.append("📦 Plugin directory: ").append(pluginDirectory).append("\n\n");
        
        // List available providers
        if (providerRegistry != null) {
            try {
                var providers = providerRegistry.getAllProviders();
                if (providers != null && !providers.isEmpty()) {
                    message.append("🔍 Available providers:\n");
                    for (var provider : providers) {
                        message.append("   - ").append(provider.id()).append("\n");
                    }
                } else {
                    message.append("🔍 No providers currently installed\n");
                }
            } catch (Exception e) {
                message.append("🔍 Unable to list providers\n");
            }
        }
        
        return message.toString();
    }

    /**
     * Get error message for missing capability.
     */
    public String getCapabilityNotAvailableError(String capability) {
        StringBuilder message = new StringBuilder();
        message.append("❌ Capability '").append(capability).append("' is not available.\n\n");
        
        String requiredPlugins = getRequiredPluginsForCapability(capability);
        if (!requiredPlugins.isEmpty()) {
            message.append("📦 Required plugins:\n");
            message.append(requiredPlugins);
            message.append("\n");
        }
        
        message.append("📦 To install:\n");
        message.append("   1. Download from: https://gollek.ai/plugins\n");
        message.append("   2. Place in: ").append(pluginDirectory).append("\n");
        message.append("   3. Restart application\n");
        
        return message.toString();
    }

    // ───────────────────────────────────────────────────────────────────────
    // Plugin Directory Management
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Ensure plugin directory exists.
     */
    public boolean ensurePluginDirectory() {
        Path pluginDir = Paths.get(pluginDirectory);
        
        if (Files.exists(pluginDir)) {
            return true;
        }

        try {
            Files.createDirectories(pluginDir);
            LOG.infof("Created plugin directory: %s", pluginDir);
            
            // Create subdirectories
            Files.createDirectories(pluginDir.resolve("runners"));
            Files.createDirectories(pluginDir.resolve("kernels"));
            Files.createDirectories(pluginDir.resolve("providers"));
            Files.createDirectories(pluginDir.resolve("optimizations"));
            Files.createDirectories(pluginDir.resolve("features"));
            
            return true;
        } catch (Exception e) {
            LOG.errorf("Failed to create plugin directory: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Get plugin directory path.
     */
    public String getPluginDirectory() {
        return pluginDirectory;
    }

    /**
     * Get deployment mode.
     */
    public DeploymentMode getDeploymentMode() {
        return deploymentMode;
    }

    /**
     * Get discovered plugins.
     */
    public Collection<PluginDescriptor> getDiscoveredPlugins() {
        return Collections.unmodifiableCollection(discoveredPlugins.values());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    private DeploymentMode detectDeploymentMode() {
        String mode = System.getProperty(DEPLOYMENT_MODE_PROPERTY, "hybrid");
        try {
            return DeploymentMode.valueOf(mode.toUpperCase());
        } catch (Exception e) {
            return DeploymentMode.HYBRID;
        }
    }
}
