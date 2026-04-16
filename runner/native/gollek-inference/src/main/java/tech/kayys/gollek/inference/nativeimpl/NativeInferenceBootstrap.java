package tech.kayys.gollek.inference.nativeimpl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import io.quarkus.runtime.StartupEvent;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.tensor.ComputeKernelRegistry;

/**
 * Bootstrap for native inference runner. Registers providers and performs lightweight
 * initialization at application startup.
 */
@ApplicationScoped
public class NativeInferenceBootstrap {
    private static final Logger log = Logger.getLogger(NativeInferenceBootstrap.class);

    void onStart(@Observes StartupEvent ev) {
        log.info("Native inference bootstrap starting");

        // Ensure compute kernel registry has best available kernel
        var kernel = ComputeKernelRegistry.get().getBestAvailable();
        log.infof("Selected compute kernel: %s (%s)", kernel.deviceType(), kernel.deviceName());

        // TODO: register native loaders / prepackers with DI or service registry
        log.info("Native inference bootstrap complete");
    }
}
