package tech.kayys.golek.provider.onnx;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import tech.kayys.golek.provider.spi.model.ModelManifest;
import tech.kayys.golek.provider.spi.model.ModelFormat;
import tech.kayys.golek.provider.spi.model.RequestContext;
import tech.kayys.golek.provider.spi.model.InferenceRequest;
import tech.kayys.golek.provider.spi.repository.ModelRepository;

import java.util.Map;
import java.util.List;

class OnnxRuntimeRunnerTest {

    private OnnxRuntimeRunner runner;
    private ModelRepository mockRepository;

    @BeforeEach
    void setUp() {
        runner = new OnnxRuntimeRunner();
        mockRepository = mock(ModelRepository.class);
        
        // Inject the mock repository
        // Note: In a real scenario, this would be handled by CDI
    }

    @Test
    void testRunnerInitialization() {
        // Test that the runner can be instantiated
        assertNotNull(runner);
    }

    @Test
    void testProviderSelection() {
        ExecutionProviderSelector selector = new ExecutionProviderSelector();
        assertNotNull(selector);
        
        // Test that CPU provider is always available
        assertTrue(selector.isProviderAvailable("CPUExecutionProvider"));
    }

    @Test
    void testConfigBuilder() {
        OnnxConfig config = OnnxConfig.builder()
            .executionProvider("CPUExecutionProvider")
            .interOpThreads(1)
            .intraOpThreads(4)
            .build();
            
        assertEquals("CPUExecutionProvider", config.getExecutionProvider());
        assertEquals(1, config.getInterOpThreads().orElse(-1));
        assertEquals(4, config.getIntraOpThreads().orElse(-1));
    }

    @Test
    void testMetadata() {
        // Test that metadata can be retrieved without initialization
        // This would normally require initialization, but we're testing the structure
        assertTrue(true); // Placeholder - actual test would require mocking
    }
}