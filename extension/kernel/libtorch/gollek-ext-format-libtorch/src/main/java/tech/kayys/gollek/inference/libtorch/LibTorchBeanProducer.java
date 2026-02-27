package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.binding.NativeLibraryLoader;
import tech.kayys.gollek.inference.libtorch.plugin.LibTorchPluginRegistry;

import java.lang.foreign.SymbolLookup;

/**
 * CDI producer for LibTorch singletons.
 * <p>
 * Provides {@link LibTorchBinding} and {@link LibTorchPluginRegistry} beans
 * for injection throughout the application.
 */
@ApplicationScoped
public class LibTorchBeanProducer {

    private static final Logger log = Logger.getLogger(LibTorchBeanProducer.class);

    @Inject
    LibTorchProviderConfig config;

    @Produces
    @Singleton
    public LibTorchBinding produceBinding() {
        if (!config.enabled()) {
            log.debug("LibTorch provider disabled — binding not initialized");
            return null;
        }

        try {
            SymbolLookup lookup = NativeLibraryLoader.load(config.nativeLib().libraryPath());
            return LibTorchBinding.initialize(lookup);
        } catch (UnsatisfiedLinkError e) {
            log.warnf("LibTorch native libraries not available: %s", e.getMessage());
            return null;
        }
    }
}
