package tech.kayys.golek.provider.gemini;


public class GeminiProvider {

    private static final String PROVIDER_ID = "gemini";

    private final ProviderCapabilities capabilities = ProviderCapabilities.builder()
            .streaming(true)
            .functionCalling(true) // Gemini supports function calling
            .multimodal(true) // Native multimodal support
            .maxContextTokens(1000000) // Gemini 1.5 Pro
            .supportedModels(
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.0-pro")
            .build();

    // Gemini uses URL parameter for API key
    // Different endpoint structure: /v1beta/models/{model
}