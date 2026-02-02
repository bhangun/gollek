package tech.kayys.golek.engine.loader;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.api.provider.ProviderConfig;
// import tech.kayys.golek.provider.core.memory.VaultSecretManager;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Loads provider configurations from multiple sources:
 * - YAML files
 * - Environment variables
 * - MicroProfile Config
 * - Vault (for secrets)
 */
@ApplicationScoped
public class ProviderConfigLoader {

    private static final Logger LOG = Logger.getLogger(ProviderConfigLoader.class);

    @ConfigProperty(name = "providers.config.path", defaultValue = "./config/providers")
    String configPath;

    @ConfigProperty(name = "providers.hot-reload.enabled", defaultValue = "true")
    boolean hotReloadEnabled;

    // @Inject
    // VaultSecretManager vaultSecretManager;

    private final Map<String, ProviderConfig> configCache = new ConcurrentHashMap<>();
    private final ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());
    private final ObjectMapper jsonMapper = new ObjectMapper();

    /**
     * Load all provider configurations
     */
    public Map<String, ProviderConfig> loadAll() {
        LOG.info("Loading provider configurations from: " + configPath);

        Map<String, ProviderConfig> configs = new HashMap<>();

        // Load from YAML files
        try {
            configs.putAll(loadFromYamlFiles());
        } catch (IOException e) {
            LOG.error("Failed to load YAML configs", e);
        }

        // Override with environment variables
        configs.putAll(loadFromEnvironment());

        // Load secrets from Vault
        configs.replaceAll((id, config) -> loadSecrets(config));

        // Cache configs
        configCache.clear();
        configCache.putAll(configs);

        LOG.infof("Loaded %d provider configurations", configs.size());
        return Collections.unmodifiableMap(configs);
    }

    /**
     * Load configuration for specific provider
     */
    public Optional<ProviderConfig> load(String providerId) {
        // Check cache first
        ProviderConfig cached = configCache.get(providerId);
        if (cached != null) {
            return Optional.of(cached);
        }

        // Load from all sources
        Map<String, ProviderConfig> all = loadAll();
        return Optional.ofNullable(all.get(providerId));
    }

    /**
     * Reload configuration for provider
     */
    public Optional<ProviderConfig> reload(String providerId) {
        configCache.remove(providerId);
        return load(providerId);
    }

    /**
     * Load from YAML files
     */
    private Map<String, ProviderConfig> loadFromYamlFiles() throws IOException {
        Map<String, ProviderConfig> configs = new HashMap<>();
        Path configDir = Paths.get(configPath);

        if (!Files.exists(configDir)) {
            LOG.warnf("Config directory does not exist: %s", configPath);
            return configs;
        }

        Files.list(configDir)
                .filter(p -> p.toString().endsWith(".yaml") ||
                        p.toString().endsWith(".yml"))
                .forEach(path -> {
                    try {
                        ProviderConfigFile file = yamlMapper.readValue(
                                path.toFile(),
                                ProviderConfigFile.class);

                        ProviderConfig config = convertToConfig(file);
                        configs.put(config.getProviderId(), config);

                        LOG.debugf("Loaded config from: %s", path);

                    } catch (IOException e) {
                        LOG.errorf(e, "Failed to load config from: %s", path);
                    }
                });

        return configs;
    }

    /**
     * Load from environment variables
     * Format: PROVIDER_{PROVIDER_ID}_{PROPERTY}
     */
    private Map<String, ProviderConfig> loadFromEnvironment() {
        Map<String, ProviderConfig> configs = new HashMap<>();
        Map<String, String> env = System.getenv();

        // Group by provider ID
        Map<String, Map<String, String>> grouped = env.entrySet().stream()
                .filter(e -> e.getKey().startsWith("PROVIDER_"))
                .collect(Collectors.groupingBy(
                        e -> extractProviderId(e.getKey()),
                        Collectors.toMap(
                                e -> extractPropertyKey(e.getKey()),
                                Map.Entry::getValue)));

        // Convert to ProviderConfig
        grouped.forEach((providerId, properties) -> {
            ProviderConfig config = ProviderConfig.builder()
                    .providerId(providerId)
                    .properties(castProperties(properties))
                    .build();

            configs.put(providerId, config);
        });

        return configs;
    }

    private String extractProviderId(String envKey) {
        // PROVIDER_OPENAI_API_KEY -> openai
        String[] parts = envKey.split("_");
        return parts.length > 1 ? parts[1].toLowerCase() : "";
    }

    private String extractPropertyKey(String envKey) {
        // PROVIDER_OPENAI_API_KEY -> api.key
        String[] parts = envKey.split("_");
        if (parts.length <= 2) {
            return "";
        }
        return String.join(".", Arrays.copyOfRange(parts, 2, parts.length))
                .toLowerCase()
                .replace('_', '.');
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> castProperties(Map<String, String> properties) {
        return new HashMap<>(properties);
    }

    /**
     * Load secrets from Vault
     */
    private ProviderConfig loadSecrets(ProviderConfig config) {
        Map<String, String> secrets = new HashMap<>();

        // Load secrets based on provider ID
        /*
         * try {
         * Map<String, String> vaultSecrets = vaultSecretManager
         * .getSecrets("providers/" + config.getProviderId());
         * 
         * secrets.putAll(vaultSecrets);
         * } catch (Exception e) {
         * LOG.warnf(e, "Failed to load secrets for provider: %s",
         * config.getProviderId());
         * }
         */

        if (secrets.isEmpty()) {
            return config;
        }

        // Merge secrets into config
        return ProviderConfig.builder()
                .providerId(config.getProviderId())
                .properties(config.getProperties())
                .secrets(secrets)
                .enabled(config.isEnabled())
                .priority(config.getPriority())
                .timeout(config.getTimeout())
                .build();
    }

    private ProviderConfig convertToConfig(ProviderConfigFile file) {
        return ProviderConfig.builder()
                .providerId(file.id)
                .properties(file.properties != null ? file.properties : Map.of())
                .enabled(file.enabled != null ? file.enabled : true)
                .priority(file.priority != null ? file.priority : 50)
                .timeout(parseDuration(file.timeout))
                .metadata("name", file.name)
                .metadata("description", file.description)
                .metadata("version", file.version)
                .build();
    }

    private Duration parseDuration(String duration) {
        if (duration == null || duration.isEmpty()) {
            return Duration.ofSeconds(30);
        }
        try {
            return Duration.parse(duration);
        } catch (Exception e) {
            LOG.warnf("Invalid duration format: %s, using default", duration);
            return Duration.ofSeconds(30);
        }
    }

    /**
     * YAML file structure
     */
    private static class ProviderConfigFile {
        public String id;
        public String name;
        public String description;
        public String version;
        public Boolean enabled;
        public Integer priority;
        public String timeout;
        public Map<String, Object> properties;
    }
}