package tech.kayys.golek.sdk.local;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.golek.engine.inference.InferenceService;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class LocalGolekSdkTest {
    
    @Mock
    private InferenceService inferenceService;
    
    @Test
    void testLocalSdkCreation() {
        // This test verifies that the LocalGolekSdk can be instantiated
        LocalGolekSdk sdk = new LocalGolekSdk();
        
        // Use reflection to inject the mocked service
        try {
            java.lang.reflect.Field serviceField = LocalGolekSdk.class.getDeclaredField("inferenceService");
            serviceField.setAccessible(true);
            serviceField.set(sdk, inferenceService);
        } catch (Exception e) {
            throw new RuntimeException("Failed to inject mock service", e);
        }
        
        // Verify that the SDK was created successfully
        org.junit.jupiter.api.Assertions.assertNotNull(sdk);
    }
}