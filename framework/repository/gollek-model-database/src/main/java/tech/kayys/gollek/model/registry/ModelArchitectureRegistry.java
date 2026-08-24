package tech.kayys.gollek.model.registry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Any;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.jboss.logging.Logger;

/**
 * CDI registry that auto-discovers all {@link ModelArchitecture}
 * implementations
 * and resolves the correct one for a given {@link ModelConfig}.
 */
@ApplicationScoped
public class ModelArchitectureRegistry {

    private static final Logger log = Logger.getLogger(ModelArchitectureRegistry.class);

    @Inject
    @Any
    Instance<ModelArchitecture> allArchitectures;

    private Iterable<ModelArchitecture> manualArchitectures;

    public Iterable<ModelArchitecture> getAllArchitectures() {
        if (allArchitectures != null) {
            return allArchitectures;
        }
        if (manualArchitectures == null) {
            List<ModelArchitecture> list = new ArrayList<>();
            java.util.ServiceLoader.load(ModelArchitecture.class).forEach(list::add);
            manualArchitectures = list;
        }
        return manualArchitectures;
    }

    /**
     * Resolve the architecture for a loaded model config.
     *
     * @param config the parsed config.json
     * @return the matching architecture
     * @throws IllegalArgumentException if no registered architecture matches
     */
    public ModelArchitecture resolve(ModelConfig config) {
        String primaryArch = config.primaryArchitecture();
        String modelType = config.modelType();

        for (ModelArchitecture arch : getAllArchitectures()) {
            if (arch.supportedArchClassNames().contains(primaryArch)) {
                log.debugf("Architecture resolved by class name: %s → %s", primaryArch, arch.id());
                return arch;
            }
        }
        for (ModelArchitecture arch : getAllArchitectures()) {
            if (modelType != null && arch.supportedModelTypes().contains(modelType)) {
                log.debugf("Architecture resolved by model_type: %s → %s", modelType, arch.id());
                return arch;
            }
        }
        throw new IllegalArgumentException(
                "No registered ModelArchitecture for arch='" + primaryArch
                        + "', model_type='" + modelType + "'. "
                        + "Register a custom ModelArchitecture CDI bean to add support.");
    }

    /**
     * Resolve the architecture for a GGUF model architecture string.
     *
     * @param ggufArch the architecture string from GGUF metadata
     * @return the matching architecture
     * @throws IllegalArgumentException if no registered architecture matches
     */
    public ModelArchitecture resolveGguf(String ggufArch) {
        for (ModelArchitecture arch : getAllArchitectures()) {
            if (arch.matchesGgufArch(ggufArch)) {
                log.debugf("Architecture resolved by GGUF arch: %s → %s", ggufArch, arch.id());
                return arch;
            }
        }
        throw new IllegalArgumentException(
                "No registered ModelArchitecture for GGUF arch='" + ggufArch + "'. "
                        + "Register a custom ModelArchitecture CDI bean to add support.");
    }

    /** All registered architecture IDs — useful for health/info endpoints. */
    public List<String> registeredIds() {
        List<String> ids = new ArrayList<>();
        getAllArchitectures().forEach(a -> ids.add(a.id()));
        return Collections.unmodifiableList(ids);
    }
}
