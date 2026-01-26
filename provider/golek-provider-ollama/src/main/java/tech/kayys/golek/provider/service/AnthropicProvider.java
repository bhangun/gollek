package tech.kayys.golek.provider.service;

public class AnthropicProvider {

    private static final String PROVIDER_ID = "anthropic";
    private static final String API_VERSION = "2023-06-01";

    // Anthropic uses different endpoint structure
    private final ProviderCapabilities capabilities = ProviderCapabilities.builder()
            .streaming(true)
            .functionCalling(false) // Claude doesn't support function calling natively
            .multimodal(true) // Claude 3 supports vision
            .maxContextTokens(200000) // Claude 3 Opus
            .supportedModels(
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307")
            .build();

    // Anthropic requires x-api-key header, not Bearer
    private String formatApiKey(String apiKey) {
        return apiKey; // No "Bearer" prefix

    }
}