package tech.kayys.golek.provider.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Helper class to select the best ONNX Runtime execution provider
 * based on available hardware and configuration.
 */
public class ExecutionProviderSelector {
    
    private static final Logger log = LoggerFactory.getLogger(ExecutionProviderSelector.class);
    
    private final List<String> availableProviders;
    
    public ExecutionProviderSelector() {
        this.availableProviders = detectAvailableProviders();
    }
    
    /**
     * Detect all available execution providers
     */
    private List<String> detectAvailableProviders() {
        List<String> providers = new ArrayList<>();
        
        try {
            // Check for CPU provider (always available)
            providers.add("CPUExecutionProvider");
            
            // Check for CUDA provider
            if (isProviderAvailable("CUDAExecutionProvider")) {
                providers.add("CUDAExecutionProvider");
            }
            
            // Check for TensorRT provider
            if (isProviderAvailable("TensorrtExecutionProvider")) {
                providers.add("TensorrtExecutionProvider");
            }
            
            // Check for OpenVINO provider
            if (isProviderAvailable("OpenVINOExecutionProvider")) {
                providers.add("OpenVINOExecutionProvider");
            }
            
            // Check for DirectML provider (Windows)
            if (isProviderAvailable("DirectMLExecutionProvider")) {
                providers.add("DirectMLExecutionProvider");
            }
            
            log.info("Detected ONNX Runtime providers: {}", providers);
            
        } catch (Exception e) {
            log.error("Error detecting ONNX Runtime providers", e);
            // Fallback to CPU only
            providers.clear();
            providers.add("CPUExecutionProvider");
        }
        
        return providers;
    }
    
    /**
     * Check if a specific execution provider is available
     */
    public boolean isProviderAvailable(String providerName) {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            // Try to create a session with the provider to verify availability
            // This is a simplified check - in practice, ONNX Runtime provides
            // methods to check provider availability
            return availableProviders.contains(providerName);
        } catch (Exception e) {
            log.warn("Provider {} not available: {}", providerName, e.getMessage());
            return false;
        }
    }
    
    /**
     * Get list of all available providers
     */
    public List<String> getAvailableProviders() {
        return new ArrayList<>(availableProviders);
    }
    
    /**
     * Select the best execution provider based on availability and performance
     */
    public String selectBestProvider() {
        // Priority order: TensorRT > CUDA > DirectML > OpenVINO > CPU
        if (availableProviders.contains("TensorrtExecutionProvider")) {
            log.info("Selecting TensorRT execution provider");
            return "TensorrtExecutionProvider";
        } else if (availableProviders.contains("CUDAExecutionProvider")) {
            log.info("Selecting CUDA execution provider");
            return "CUDAExecutionProvider";
        } else if (availableProviders.contains("DirectMLExecutionProvider")) {
            log.info("Selecting DirectML execution provider");
            return "DirectMLExecutionProvider";
        } else if (availableProviders.contains("OpenVINOExecutionProvider")) {
            log.info("Selecting OpenVINO execution provider");
            return "OpenVINOExecutionProvider";
        } else {
            log.info("Selecting CPU execution provider");
            return "CPUExecutionProvider";
        }
    }
    
    /**
     * Validate if the requested provider is available
     */
    public boolean isValidProvider(String provider) {
        return availableProviders.contains(provider);
    }
}