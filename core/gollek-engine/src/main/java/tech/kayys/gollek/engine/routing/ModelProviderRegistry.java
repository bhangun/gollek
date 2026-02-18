package tech.kayys.gollek.engine.routing;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.routing.ModelProviderMapping;
import tech.kayys.gollek.spi.routing.ModelProviderMapping.ModelType;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Registry for model-to-provider mappings.
 * Tracks which models are available on which providers.
 */
@ApplicationScoped
public class ModelProviderRegistry {

    private static final Logger LOG = Logger.getLogger(ModelProviderRegistry.class);

    // Model ID -> Mapping
    private final Map<String, ModelProviderMapping> mappings = new ConcurrentHashMap<>();

    // Provider ID -> Set of model IDs
    private final Map<String, Set<String>> providerModels = new ConcurrentHashMap<>();

    public ModelProviderRegistry() {
        initializeDefaultMappings();
    }

    /**
     * Initialize default model-provider mappings
     */
    private void initializeDefaultMappings() {
        // OpenAI models
        register(ModelProviderMapping.builder()
                .modelId("gpt-4")
                .displayName("GPT-4")
                .providers("openai", "azure-openai")
                .type(ModelType.CHAT)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("gpt-4-turbo")
                .displayName("GPT-4 Turbo")
                .providers("openai", "azure-openai")
                .type(ModelType.CHAT)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("gpt-3.5-turbo")
                .displayName("GPT-3.5 Turbo")
                .providers("openai", "azure-openai")
                .type(ModelType.CHAT)
                .build());

        // Anthropic models
        register(ModelProviderMapping.builder()
                .modelId("claude-3-opus")
                .displayName("Claude 3 Opus")
                .providers("anthropic")
                .type(ModelType.CHAT)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("claude-3-sonnet")
                .displayName("Claude 3 Sonnet")
                .providers("anthropic")
                .type(ModelType.CHAT)
                .build());

        // Google models
        register(ModelProviderMapping.builder()
                .modelId("gemini-pro")
                .displayName("Gemini Pro")
                .providers("gemini")
                .type(ModelType.CHAT)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("gemini-ultra")
                .displayName("Gemini Ultra")
                .providers("gemini")
                .type(ModelType.MULTIMODAL)
                .build());

        // Local models (available on multiple local providers)
        register(ModelProviderMapping.builder()
                .modelId("llama-3-8b")
                .displayName("Llama 3 8B")
                .providers("ollama", "local", "local-vllm")
                .type(ModelType.CHAT)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("llama-3-70b")
                .displayName("Llama 3 70B")
                .providers("local-vllm") // Large model, only on vLLM
                .type(ModelType.CHAT)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("mistral-7b")
                .displayName("Mistral 7B")
                .providers("ollama", "local", "local-vllm")
                .type(ModelType.CHAT)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("codellama-13b")
                .displayName("Code Llama 13B")
                .providers("ollama", "local")
                .type(ModelType.CODE)
                .build());

        register(ModelProviderMapping.builder()
                .modelId("phi-3")
                .displayName("Phi-3")
                .providers("ollama", "local")
                .type(ModelType.CHAT)
                .build());

        LOG.infof("Initialized %d default model mappings", mappings.size());
    }

    /**
     * Register a model-provider mapping
     */
    public void register(ModelProviderMapping mapping) {
        mappings.put(mapping.modelId(), mapping);

        // Update reverse index
        for (String providerId : mapping.providerIds()) {
            providerModels
                    .computeIfAbsent(providerId, k -> ConcurrentHashMap.newKeySet())
                    .add(mapping.modelId());
        }

        LOG.debugf("Registered model %s on providers: %s",
                mapping.modelId(), mapping.providerIds());
    }

    /**
     * Get mapping for a model
     */
    public Optional<ModelProviderMapping> getMapping(String modelId) {
        return Optional.ofNullable(mappings.get(modelId));
    }

    /**
     * Get providers for a model
     */
    public List<String> getProvidersForModel(String modelId) {
        ModelProviderMapping mapping = mappings.get(modelId);
        return mapping != null ? mapping.providerIds() : Collections.emptyList();
    }

    /**
     * Get models available on a provider
     */
    public Set<String> getModelsForProvider(String providerId) {
        return providerModels.getOrDefault(providerId, Collections.emptySet());
    }

    /**
     * Check if model is available on provider
     */
    public boolean isModelAvailable(String modelId, String providerId) {
        ModelProviderMapping mapping = mappings.get(modelId);
        return mapping != null && mapping.isAvailableOn(providerId);
    }

    /**
     * Get preferred provider for a model
     */
    public Optional<String> getPreferredProvider(String modelId) {
        return getMapping(modelId)
                .filter(ModelProviderMapping::hasPreferredProvider)
                .map(ModelProviderMapping::preferredProvider);
    }

    /**
     * Get models by type
     */
    public List<ModelProviderMapping> getModelsByType(ModelType type) {
        return mappings.values().stream()
                .filter(m -> m.type() == type)
                .collect(Collectors.toList());
    }

    /**
     * Get models available on local providers
     */
    public List<ModelProviderMapping> getLocalModels() {
        Set<String> localProviders = Set.of("ollama", "local", "local-vllm", "litert");
        return mappings.values().stream()
                .filter(m -> m.providerIds().stream().anyMatch(localProviders::contains))
                .collect(Collectors.toList());
    }

    /**
     * Get models available on cloud providers
     */
    public List<ModelProviderMapping> getCloudModels() {
        Set<String> cloudProviders = Set.of("openai", "anthropic", "gemini", "azure-openai");
        return mappings.values().stream()
                .filter(m -> m.providerIds().stream().anyMatch(cloudProviders::contains))
                .collect(Collectors.toList());
    }

    /**
     * Get all registered models
     */
    public Collection<ModelProviderMapping> getAllMappings() {
        return Collections.unmodifiableCollection(mappings.values());
    }

    /**
     * Get all provider IDs that have models
     */
    public Set<String> getAllProviderIds() {
        return Collections.unmodifiableSet(providerModels.keySet());
    }

    /**
     * Remove a model mapping
     */
    public void unregister(String modelId) {
        ModelProviderMapping removed = mappings.remove(modelId);
        if (removed != null) {
            for (String providerId : removed.providerIds()) {
                Set<String> models = providerModels.get(providerId);
                if (models != null) {
                    models.remove(modelId);
                }
            }
        }
    }

    /**
     * Clear all mappings
     */
    public void clear() {
        mappings.clear();
        providerModels.clear();
    }
}
