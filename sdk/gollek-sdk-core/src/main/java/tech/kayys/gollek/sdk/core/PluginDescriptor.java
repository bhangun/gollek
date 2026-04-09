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

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.Attributes;
import java.util.jar.Manifest;

/**
 * Plugin descriptor with manifest-based metadata.
 *
 * <p>Plugins declare their capabilities, dependencies, and metadata via JAR manifest entries.
 * This makes the plugin system flexible, agnostic, and extensible without hardcoded mappings.</p>
 *
 * <h2>Manifest Entries</h2>
 * <pre>
 * Plugin-Id: gguf-runner
 * Plugin-Type: runner
 * Plugin-Version: 2.0.0
 * Plugin-Name: GGUF Runner
 * Plugin-Description: GGUF format support using llama.cpp
 * Plugin-Capabilities: gguf-inference, llama-architecture, mistral-architecture
 * Plugin-Dependencies: kernel-plugin (optional)
 * Plugin-Provider: tech.kayys.gollek.plugin.runner.gguf.GGUFRunnerPlugin
 * Plugin-Deployment: standalone,microservice
 * </pre>
 *
 * @since 2.0.0
 */
public class PluginDescriptor {

    private final String id;
    private final String type;
    private final String version;
    private final String name;
    private final String description;
    private final Set<String> capabilities;
    private final Set<String> dependencies;
    private final String providerClass;
    private final Set<DeploymentMode> supportedDeployments;
    private final Map<String, String> metadata;

    /**
     * Create plugin descriptor from manifest.
     *
     * @param manifest JAR manifest
     */
    public PluginDescriptor(Manifest manifest) {
        Attributes attrs = manifest.getMainAttributes();
        
        this.id = attrs.getValue("Plugin-Id");
        this.type = attrs.getValue("Plugin-Type");
        this.version = Optional.ofNullable(attrs.getValue("Plugin-Version")).orElse("1.0.0");
        this.name = Optional.ofNullable(attrs.getValue("Plugin-Name")).orElse(id);
        this.description = Optional.ofNullable(attrs.getValue("Plugin-Description")).orElse("");
        this.capabilities = parseCapabilities(attrs.getValue("Plugin-Capabilities"));
        this.dependencies = parseDependencies(attrs.getValue("Plugin-Dependencies"));
        this.providerClass = attrs.getValue("Plugin-Provider");
        this.supportedDeployments = parseDeployments(attrs.getValue("Plugin-Deployment"));
        this.metadata = extractMetadata(attrs);
    }

    /**
     * Create plugin descriptor with builder.
     */
    private PluginDescriptor(Builder builder) {
        this.id = builder.id;
        this.type = builder.type;
        this.version = builder.version;
        this.name = builder.name;
        this.description = builder.description;
        this.capabilities = builder.capabilities;
        this.dependencies = builder.dependencies;
        this.providerClass = builder.providerClass;
        this.supportedDeployments = builder.supportedDeployments;
        this.metadata = builder.metadata;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Getters
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Get plugin ID.
     */
    public String getId() {
        return id;
    }

    /**
     * Get plugin type.
     */
    public String getType() {
        return type;
    }

    /**
     * Get plugin version.
     */
    public String getVersion() {
        return version;
    }

    /**
     * Get plugin name.
     */
    public String getName() {
        return name;
    }

    /**
     * Get plugin description.
     */
    public String getDescription() {
        return description;
    }

    /**
     * Get plugin capabilities.
     */
    public Set<String> getCapabilities() {
        return Collections.unmodifiableSet(capabilities);
    }

    /**
     * Check if plugin has specific capability.
     */
    public boolean hasCapability(String capability) {
        return capabilities.contains(capability);
    }

    /**
     * Get plugin dependencies.
     */
    public Set<String> getDependencies() {
        return Collections.unmodifiableSet(dependencies);
    }

    /**
     * Get provider class name.
     */
    public String getProviderClass() {
        return providerClass;
    }

    /**
     * Get supported deployment modes.
     */
    public Set<DeploymentMode> getSupportedDeployments() {
        return Collections.unmodifiableSet(supportedDeployments);
    }

    /**
     * Check if plugin supports deployment mode.
     */
    public boolean supportsDeployment(DeploymentMode mode) {
        return supportedDeployments.contains(mode);
    }

    /**
     * Get metadata.
     */
    public Map<String, String> getMetadata() {
        return Collections.unmodifiableMap(metadata);
    }

    /**
     * Get metadata value.
     */
    public String getMetadata(String key) {
        return metadata.get(key);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Parsing Methods
    // ───────────────────────────────────────────────────────────────────────

    private Set<String> parseCapabilities(String value) {
        Set<String> capabilities = new HashSet<>();
        if (value != null && !value.trim().isEmpty()) {
            Arrays.stream(value.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .forEach(capabilities::add);
        }
        return capabilities;
    }

    private Set<String> parseDependencies(String value) {
        Set<String> dependencies = new HashSet<>();
        if (value != null && !value.trim().isEmpty()) {
            Arrays.stream(value.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .forEach(dependencies::add);
        }
        return dependencies;
    }

    private Set<DeploymentMode> parseDeployments(String value) {
        Set<DeploymentMode> modes = new HashSet<>();
        if (value != null && !value.trim().isEmpty()) {
            Arrays.stream(value.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> {
                    try {
                        return DeploymentMode.valueOf(s.toUpperCase());
                    } catch (Exception e) {
                        return DeploymentMode.HYBRID; // Default
                    }
                })
                .forEach(modes::add);
        } else {
            modes.add(DeploymentMode.HYBRID); // Default supports all
        }
        return modes;
    }

    private Map<String, String> extractMetadata(Attributes attrs) {
        Map<String, String> metadata = new HashMap<>();
        
        // Extract common metadata
        if (attrs.containsKey("Plugin-Author")) {
            metadata.put("author", attrs.getValue("Plugin-Author"));
        }
        if (attrs.containsKey("Plugin-Vendor")) {
            metadata.put("vendor", attrs.getValue("Plugin-Vendor"));
        }
        if (attrs.containsKey("Plugin-License")) {
            metadata.put("license", attrs.getValue("Plugin-License"));
        }
        if (attrs.containsKey("Plugin-Homepage")) {
            metadata.put("homepage", attrs.getValue("Plugin-Homepage"));
        }
        if (attrs.containsKey("Plugin-Documentation")) {
            metadata.put("documentation", attrs.getValue("Plugin-Documentation"));
        }
        if (attrs.containsKey("Plugin-Repository")) {
            metadata.put("repository", attrs.getValue("Plugin-Repository"));
        }
        
        // Extract GPU requirements
        if (attrs.containsKey("Plugin-GPU-Requirement")) {
            metadata.put("gpu_requirement", attrs.getValue("Plugin-GPU-Requirement"));
        }
        if (attrs.containsKey("Plugin-Minimum-Compute-Capability")) {
            metadata.put("min_compute_capability", attrs.getValue("Plugin-Minimum-Compute-Capability"));
        }
        if (attrs.containsKey("Plugin-Minimum-Memory")) {
            metadata.put("min_memory", attrs.getValue("Plugin-Minimum-Memory"));
        }
        
        // Extract performance metadata
        if (attrs.containsKey("Plugin-Performance-Speedup")) {
            metadata.put("speedup", attrs.getValue("Plugin-Performance-Speedup"));
        }
        if (attrs.containsKey("Plugin-Performance-Memory-Overhead")) {
            metadata.put("memory_overhead", attrs.getValue("Plugin-Performance-Memory-Overhead"));
        }
        
        return metadata;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Builder
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Create builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for PluginDescriptor.
     */
    public static class Builder {
        private String id;
        private String type;
        private String version = "1.0.0";
        private String name;
        private String description = "";
        private Set<String> capabilities = new HashSet<>();
        private Set<String> dependencies = new HashSet<>();
        private String providerClass;
        private Set<DeploymentMode> supportedDeployments = EnumSet.of(DeploymentMode.HYBRID);
        private Map<String, String> metadata = new HashMap<>();

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder capabilities(String... capabilities) {
            this.capabilities.addAll(Arrays.asList(capabilities));
            return this;
        }

        public Builder capability(String capability) {
            this.capabilities.add(capability);
            return this;
        }

        public Builder dependencies(String... dependencies) {
            this.dependencies.addAll(Arrays.asList(dependencies));
            return this;
        }

        public Builder providerClass(String providerClass) {
            this.providerClass = providerClass;
            return this;
        }

        public Builder supportedDeployments(DeploymentMode... modes) {
            this.supportedDeployments = EnumSet.copyOf(Arrays.asList(modes));
            return this;
        }

        public Builder metadata(String key, String value) {
            this.metadata.put(key, value);
            return this;
        }

        public PluginDescriptor build() {
            if (id == null || type == null) {
                throw new IllegalArgumentException("Plugin ID and Type are required");
            }
            if (name == null) {
                name = id;
            }
            return new PluginDescriptor(this);
        }
    }

    @Override
    public String toString() {
        return String.format(
            "PluginDescriptor{id='%s', type='%s', version='%s', capabilities=%s}",
            id, type, version, capabilities
        );
    }
}
