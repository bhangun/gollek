package tech.kayys.golek.engine.module;

import tech.kayys.golek.core.config.ConfigurationManager;
import tech.kayys.golek.core.plugin.PluginManager;
import tech.kayys.golek.spi.provider.ProviderRegistry;
import tech.kayys.golek.engine.plugin.GolekPluginRegistry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import org.jboss.logging.Logger;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;

/**
 * Central kernel module that orchestrates all core components
 * Provides a unified entry point for the inference platform
 */
@ApplicationScoped
public class KernelModule {

    private static final Logger LOG = Logger.getLogger(KernelModule.class);

    private final PluginManager pluginManager;
    private final ConfigurationManager configurationManager;
    private final ProviderRegistry providerRegistry;
    private final GolekPluginRegistry kernelPluginRegistry;

    private volatile boolean initialized = false;
    private volatile boolean started = false;

    public KernelModule(ProviderRegistry providerRegistry) {
        // Initialize OpenTelemetry SDK
        SdkTracerProvider tracerProvider = SdkTracerProvider.builder().build();
        SdkMeterProvider meterProvider = SdkMeterProvider.builder().build();

        OpenTelemetry openTelemetry = OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .setMeterProvider(meterProvider)
                .build();

        // Initialize core components
        this.configurationManager = new ConfigurationManager();
        this.pluginManager = new PluginManager();
        this.providerRegistry = providerRegistry;
        this.kernelPluginRegistry = new GolekPluginRegistry();
    }

    /**
     * Initialize the kernel module
     */
    public void initialize() {
        if (initialized) {
            LOG.warn("Kernel module already initialized");
            return;
        }

        LOG.info("Initializing kernel module");

        try {
            // Initialize configuration manager
            LOG.debug("Initializing configuration manager");

            // Initialize plugin manager
            LOG.debug("Initializing plugin manager");
            pluginManager.initialize();

            // Initialize kernel plugin registry
            LOG.debug("Initializing kernel plugin registry");
            kernelPluginRegistry.initialize();

            // Initialize provider registry
            LOG.debug("Initializing provider registry");

            initialized = true;
            LOG.info("Kernel module initialized successfully");

        } catch (Exception e) {
            LOG.error("Failed to initialize kernel module", e);
            throw new RuntimeException("Kernel initialization failed", e);
        }
    }

    /**
     * Start the kernel module
     */
    public void start() {
        if (!initialized) {
            throw new IllegalStateException("Kernel module not initialized");
        }

        if (started) {
            LOG.warn("Kernel module already started");
            return;
        }

        LOG.info("Starting kernel module");

        try {
            // Start plugin manager
            LOG.debug("Starting plugin manager");
            pluginManager.start();

            started = true;
            LOG.info("Kernel module started successfully");

        } catch (Exception e) {
            LOG.error("Failed to start kernel module", e);
            throw new RuntimeException("Kernel startup failed", e);
        }
    }

    /**
     * Stop the kernel module
     */
    public void stop() {
        if (!started) {
            LOG.warn("Kernel module not started");
            return;
        }

        LOG.info("Stopping kernel module");

        try {
            // Stop plugin manager
            LOG.debug("Stopping plugin manager");
            pluginManager.stop();

            // Shutdown kernel plugin registry
            LOG.debug("Shutting down kernel plugin registry");
            kernelPluginRegistry.shutdownAll();

            started = false;
            LOG.info("Kernel module stopped successfully");

        } catch (Exception e) {
            LOG.error("Failed to stop kernel module", e);
            throw new RuntimeException("Kernel shutdown failed", e);
        }
    }

    /**
     * Get the plugin manager
     */
    @Produces
    @Singleton
    public PluginManager getPluginManager() {
        return pluginManager;
    }

    /**
     * Get the configuration manager
     */
    @Produces
    @Singleton
    public ConfigurationManager getConfigurationManager() {
        return configurationManager;
    }


    /**
     * Get the enhanced provider registry
     */
    public ProviderRegistry getProviderRegistry() {
        return providerRegistry;
    }

    /**
     * Get the kernel plugin registry
     */
    public GolekPluginRegistry getKernelPluginRegistry() {
        return kernelPluginRegistry;
    }

    /**
     * Check if kernel is initialized
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Check if kernel is started
     */
    public boolean isStarted() {
        return started;
    }

    /**
     * Get kernel module status
     */
    public KernelStatus getStatus() {
        return new KernelStatus(initialized, started);
    }

    /**
     * Kernel status record
     */
    public record KernelStatus(boolean initialized, boolean started) {
    }
}