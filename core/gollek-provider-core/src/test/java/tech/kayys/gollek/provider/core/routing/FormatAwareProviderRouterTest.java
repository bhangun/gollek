package tech.kayys.gollek.provider.core.routing;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.inject.Instance;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.provider.*;
import tech.kayys.gollek.spi.stream.StreamChunk;
import tech.kayys.gollek.provider.core.registry.DefaultLocalModelRegistry;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link FormatAwareProviderRouter} — provider selection
 * based on model format detection.
 *
 * <p>Uses reflection to inject mock providers and a real
 * {@link DefaultLocalModelRegistry} (no CDI container needed).
 */
class FormatAwareProviderRouterTest {

    @TempDir
    Path tempDir;

    private FormatAwareProviderRouter router;
    private DefaultLocalModelRegistry registry;
    private List<StreamingProvider> mockProviders;

    @BeforeEach
    void setUp() throws Exception {
        router = new FormatAwareProviderRouter();
        registry = new DefaultLocalModelRegistry();
        mockProviders = new ArrayList<>();

        // Inject registry via reflection
        setField(router, "localModelRegistry", registry);
    }

    // ── Route by registry format ───────────────────────────────────────────

    @Test
    void routesToGgufProviderWhenRegistryResolvesToGguf() throws Exception {
        // Register a GGUF model in the registry
        Path gguf = createGgufFile("llama.gguf");
        registry.register("llama", gguf, ModelFormat.GGUF);

        // Add mock GGUF provider
        StubProvider ggufProvider = new StubProvider("gguf",
                Set.of(ModelFormat.GGUF), true);
        StubProvider stProvider = new StubProvider("safetensor",
                Set.of(ModelFormat.SAFETENSORS), true);
        mockProviders.addAll(List.of(ggufProvider, stProvider));
        injectProviders();

        ProviderRequest request = ProviderRequest.builder()
                .model("llama")
                .message(Message.user("Hello"))
                .build();

        InferenceResponse resp = router.route(request).await().indefinitely();

        assertEquals("gguf", resp.getModel(),
                "Should route to GGUF provider when registry entry is GGUF");
    }

    @Test
    void routesToSafeTensorProviderWhenRegistryResolvesToSafeTensors() throws Exception {
        Path st = createSafeTensorsFile("bert.safetensors");
        registry.register("bert", st, ModelFormat.SAFETENSORS);

        StubProvider ggufProvider = new StubProvider("gguf",
                Set.of(ModelFormat.GGUF), true);
        StubProvider stProvider = new StubProvider("safetensor",
                Set.of(ModelFormat.SAFETENSORS), true);
        mockProviders.addAll(List.of(ggufProvider, stProvider));
        injectProviders();

        ProviderRequest request = ProviderRequest.builder()
                .model("bert")
                .message(Message.user("Embed this"))
                .build();

        InferenceResponse resp = router.route(request).await().indefinitely();

        assertEquals("safetensor", resp.getModel(),
                "Should route to SafeTensor provider");
    }

    // ── fallback when no format-specific provider ──────────────────────────

    @Test
    void fallsBackToGenericSupportWhenNoFormatMatch() throws Exception {
        // Register model with ONNX format — no ONNX provider exists
        Path gguf = createGgufFile("model.gguf");
        registry.register("model", gguf, ModelFormat.ONNX); // mismatch intentional

        // Only GGUF provider, which doesn't support ONNX format
        // but supports() returns true for any model
        StubProvider universalProvider = new StubProvider("universal",
                Set.of(), true); // supports all models, no format filter
        mockProviders.add(universalProvider);
        injectProviders();

        ProviderRequest request = ProviderRequest.builder()
                .model("model")
                .message(Message.user("Hello"))
                .build();

        InferenceResponse resp = router.route(request).await().indefinitely();

        assertEquals("universal", resp.getModel(),
                "Should fall back to generic provider when no format match");
    }

    // ── Throws when no provider found ──────────────────────────────────────

    @Test
    void throwsWhenNoProviderSupportsModel() throws Exception {
        StubProvider picky = new StubProvider("picky",
                Set.of(ModelFormat.GGUF), false); // supports() returns false
        mockProviders.add(picky);
        injectProviders();

        ProviderRequest request = ProviderRequest.builder()
                .model("unknown-model")
                .message(Message.user("Hello"))
                .build();

        assertThrows(Exception.class,
                () -> router.route(request).await().indefinitely(),
                "Should throw when no provider supports the model");
    }

    // ── resolveFormat ──────────────────────────────────────────────────────

    @Test
    void resolveFormatFromRegistry() throws IOException {
        Path gguf = createGgufFile("model.gguf");
        registry.register("my-model", gguf, ModelFormat.GGUF);

        Optional<ModelFormat> fmt = router.resolveFormat("my-model");
        assertTrue(fmt.isPresent());
        assertEquals(ModelFormat.GGUF, fmt.get());
    }

    @Test
    void resolveFormatByExtension() {
        Optional<ModelFormat> fmt = router.resolveFormat("model.safetensors");
        assertTrue(fmt.isPresent());
        assertEquals(ModelFormat.SAFETENSORS, fmt.get());
    }

    @Test
    void resolveFormatReturnsEmptyForUnknown() {
        Optional<ModelFormat> fmt = router.resolveFormat("completely-unknown");
        assertTrue(fmt.isEmpty());
    }

    // ── Priority ordering ──────────────────────────────────────────────────

    @Test
    void respectsProviderPriority() throws Exception {
        Path gguf = createGgufFile("model.gguf");
        registry.register("model", gguf, ModelFormat.GGUF);

        // Both providers support GGUF, but "gguf" has higher priority
        StubProvider ggufPrimary = new StubProvider("gguf",
                Set.of(ModelFormat.GGUF), true);
        StubProvider ggufSecondary = new StubProvider("libtorch",
                Set.of(ModelFormat.GGUF), true);
        mockProviders.addAll(List.of(ggufSecondary, ggufPrimary)); // add in wrong order
        injectProviders();

        ProviderRequest request = ProviderRequest.builder()
                .model("model")
                .message(Message.user("Hi"))
                .build();

        InferenceResponse resp = router.route(request).await().indefinitely();

        assertEquals("gguf", resp.getModel(),
                "Should pick 'gguf' provider over 'libtorch' due to priority ordering");
    }

    // ── Streaming ──────────────────────────────────────────────────────────

    @Test
    void routeStreamGoesToCorrectProvider() throws Exception {
        Path st = createSafeTensorsFile("bert.safetensors");
        registry.register("bert", st, ModelFormat.SAFETENSORS);

        StubProvider stProvider = new StubProvider("safetensor",
                Set.of(ModelFormat.SAFETENSORS), true);
        mockProviders.add(stProvider);
        injectProviders();

        ProviderRequest request = ProviderRequest.builder()
                .model("bert")
                .message(Message.user("Stream this"))
                .streaming(true)
                .build();

        List<StreamChunk> chunks = router.routeStream(request)
                .collect().asList().await().indefinitely();

        assertFalse(chunks.isEmpty(), "Should receive stream chunks");
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private Path createGgufFile(String name) throws IOException {
        Path file = tempDir.resolve(name);
        byte[] header = new byte[16];
        ByteBuffer.wrap(header, 0, 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(0x46554747);
        Files.write(file, header);
        return file;
    }

    private Path createSafeTensorsFile(String name) throws IOException {
        Path file = tempDir.resolve(name);
        byte[] data = new byte[32];
        ByteBuffer.wrap(data, 0, 8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(64L);
        data[8] = '{';
        Files.write(file, data);
        return file;
    }

    private void injectProviders() throws Exception {
        // Create a simple Instance<StreamingProvider> wrapper
        Instance<StreamingProvider> instance = new SimpleInstance<>(mockProviders);
        setField(router, "providers", instance);
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    // ── Stub provider ──────────────────────────────────────────────────────

    /**
     * Minimal StreamingProvider stub that returns a response containing
     * its own ID as the model name (so tests can verify which provider was chosen).
     */
    static class StubProvider implements StreamingProvider {

        private final String providerId;
        private final Set<ModelFormat> formats;
        private final boolean supportsAll;

        StubProvider(String id, Set<ModelFormat> formats, boolean supportsAll) {
            this.providerId = id;
            this.formats = formats;
            this.supportsAll = supportsAll;
        }

        @Override
        public String id() { return providerId; }

        @Override
        public String name() { return providerId; }

        @Override
        public ProviderMetadata metadata() {
            return ProviderMetadata.builder()
                    .providerId(providerId)
                    .name(providerId)
                    .version("1.0.0")
                    .build();
        }

        @Override
        public void initialize(ProviderConfig config) {
            // No-op for stub
        }

        @Override
        public boolean supports(String modelId, ProviderRequest request) {
            return supportsAll;
        }

        @Override
        public ProviderCapabilities capabilities() {
            return ProviderCapabilities.builder()
                    .supportedFormats(formats)
                    .build();
        }

        @Override
        public Uni<ProviderHealth> health() {
            return Uni.createFrom().item(ProviderHealth.healthy());
        }

        @Override
        public void shutdown() {
            // No-op for stub
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request) {
            return Uni.createFrom().item(InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(providerId) // encode provider id in model field for assertion
                    .content("Response from " + providerId)
                    .durationMs(10)
                    .build());
        }

        @Override
        public Multi<StreamChunk> inferStream(ProviderRequest request) {
            return Multi.createFrom().items(
                    StreamChunk.of(request.getRequestId(), 0, "chunk from " + providerId)
            );
        }
    }

    /**
     * Minimal CDI Instance implementation for testing.
     */
    static class SimpleInstance<T> implements Instance<T> {
        private final List<T> items;
        SimpleInstance(List<T> items) { this.items = items; }

        @Override public Iterator<T> iterator() { return items.iterator(); }
        @Override public Instance<T> select(java.lang.annotation.Annotation... qualifiers) { return this; }
        @Override public <U extends T> Instance<U> select(Class<U> subtype, java.lang.annotation.Annotation... qualifiers) { throw new UnsupportedOperationException(); }
        @Override public <U extends T> Instance<U> select(jakarta.enterprise.util.TypeLiteral<U> subtype, java.lang.annotation.Annotation... qualifiers) { throw new UnsupportedOperationException(); }
        @Override public boolean isUnsatisfied() { return items.isEmpty(); }
        @Override public boolean isAmbiguous() { return items.size() > 1; }
        @Override public boolean isResolvable() { return items.size() == 1; }
        @Override public T get() { return items.get(0); }
        @Override public void destroy(T instance) {}
        @Override public Handle<T> getHandle() { throw new UnsupportedOperationException(); }
        @Override public Iterable<Handle<T>> handles() { throw new UnsupportedOperationException(); }
    }
}
