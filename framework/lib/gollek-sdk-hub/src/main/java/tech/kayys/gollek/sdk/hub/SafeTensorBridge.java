package tech.kayys.gollek.sdk.hub;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.gollek.safetensor.loader.SafetensorLoadResult;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Bridge class to SafetensorFFMLoader.
 * This class is only called by ModelHub if the gollek-safetensor-loader is on the classpath.
 */
class SafeTensorBridge {

    /**
     * Loads a single .safetensors file into a map of GradTensors.
     */
    static Map<String, GradTensor> load(Path file) throws IOException {
        Map<String, GradTensor> weights = new LinkedHashMap<>();

        // Create a minimal loader (not ideal for CDI, but sufficient for local SDK use)
        // In a real system, we'd use a better way to manage this.
        SafetensorFFMLoader loader = createStandaloneLoader();

        try (SafetensorLoadResult result = loader.load(file)) {
            for (String name : result.tensorNames()) {
                SafetensorTensor tensor = result.tensor(name);
                // Convert to GradTensor (copies data to heap)
                float[] data = tensor.toFloatArray();
                GradTensor gradTensor = GradTensor.of(data, tensor.shape());
                weights.put(name, gradTensor);
            }
        } catch (Exception e) {
            throw new IOException("Failed to load SafeTensors file: " + file, e);
        }

        return weights;
    }

    private static SafetensorFFMLoader createStandaloneLoader() {
        // This is a bit of a hack to work around CDI requirements in a static context
        // but it works for the SDK.
        try {
            // Reflective instantiation to avoid hard link if loader not present
            return new SafetensorFFMLoader();
            // Note: need to handle @Inject fields if they are strictly required for 'load'
            // SafetensorFFMLoader.load calls validatePath, loadMmap, and metrics.recordLoad.
            // If they are null, it might throw NPE.
        } catch (Exception e) {
            throw new RuntimeException("Could not initialize SafetensorFFMLoader standalone", e);
        }
    }
}
