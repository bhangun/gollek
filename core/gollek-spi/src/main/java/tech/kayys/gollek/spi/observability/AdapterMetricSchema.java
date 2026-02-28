package tech.kayys.gollek.spi.observability;

/**
 * Shared metric names and tags for adapter-aware inference providers.
 * <p>
 * Providers may emit additional provider-specific metrics, but these names
 * should remain consistent across runtimes (gguf/libtorch/onnx/tflite/...).
 */
public final class AdapterMetricSchema {

    private AdapterMetricSchema() {
    }

    // Metric names
    public static final String REQUEST_TOTAL = "adapter.request.total";
    public static final String SESSION_ACQUIRE_DURATION = "adapter.session.acquire.duration";
    public static final String INIT_WAIT_DURATION = "adapter.init.wait.duration";
    public static final String APPLY_DURATION = "adapter.apply.duration";
    public static final String CACHE_HIT = "adapter.cache.hit";
    public static final String CACHE_MISS = "adapter.cache.miss";

    // Tag keys
    public static final String TAG_PROVIDER = "provider";
    public static final String TAG_TYPE = "type";
    public static final String TAG_SCOPE = "scope";

    // Common tag values
    public static final String TYPE_UNKNOWN = "unknown";
    public static final String SCOPE_PAIR_INDEX = "pair_index";
    public static final String SCOPE_ADAPTER_REGISTRY = "adapter_registry";
}
