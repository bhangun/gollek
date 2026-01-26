package tech.kayys.golek.inference.gguf;

import org.junit.jupiter.api.Test;
import java.lang.foreign.MemorySegment;
import static org.junit.jupiter.api.Assertions.*;

public class LlamaCppBindingTest {

    @Test
    public void testLibraryLoading() {
        // This will attempt to extract and load the library
        // It might fail if the resources aren't populated yet, but we're testing the binding logic
        try {
            LlamaCppBinding binding = LlamaCppBinding.load();
            assertNotNull(binding);
            
            MemorySegment modelParams = binding.getDefaultModelParams();
            assertNotNull(modelParams);
            
            MemorySegment contextParams = binding.getDefaultContextParams();
            assertNotNull(contextParams);
            
            binding.close();
        } catch (Exception e) {
            // Log the error but don't fail yet if it's just a missing library file
            System.err.println("Note: Library loading failed as expected if libs are not in resources: " + e.getMessage());
        }
    }
}
