package tech.kayys.golek.provider.cerebras;

public class CerebrasProvider {

    private static final Logger LOG = Logger.getLogger(CerebrasProvider.class);
    private static final String PROVIDER_ID = "cerebras";

    @Inject
    CerebrasClient client;

    @Inject
    CerebrasConfig config;

    @Inject
    RequestMapper requestMapper;

    @Inject
    ResponseMapper responseMapper;

    @Inject
    MeterRegistry meterRegistry;

    @Inject
    Tracer tracer;

    private final ProviderCapabilities capabilities = ProviderCapabilities.builder()
            .streaming(true)
            .functionCalling(false) // Cerebras doesn't support function calling yet
            .multimodal(false) // Text-only for now
            .maxContextTokens(131072) // 128K context window
            .supportedModels(
                    "llama-3.3-70b",
                    "llama-3.1-70b",
                    "llama-3.1-8b")
            .metadata("provider_type", "ultra_fast")
            .metadata("hardware", "wafer_scale_engine")
            .build();

    @Override
    public String id() {
        return PROVIDER_ID;
    }

}