package tech.kayys.golek.core.module;

import tech.kayys.golek.core.config.EnhancedConfigurationManager;
import tech.kayys.golek.core.plugin.GolekPluginRegistry;
import tech.kayys.golek.core.plugin.PluginManager;
import tech.kayys.golek.core.reliability.ReliabilityManager;
import tech.kayys.golek.kernel.KernelModule;
import tech.kayys.golek.kernel.observability.ObservabilityManager;
import tech.kayys.golek.kernel.provider.EnhancedProviderRegistry;

import org.jboss.logging.Logger;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Main system module that orchestrates all core components
 * Designed for modularity, extensibility, and reliability
 */
@ApplicationScoped
public class SystemModule {

    private static final Logger LOG = Logger.getLogger(SystemModule.class);

    private final KernelModule kernelModule;
    private final PluginManager pluginManager;
    private final EnhancedConfigurationManager configurationManager;
    private final ObservabilityManager observabilityManager;
    private final EnhancedProviderRegistry providerRegistry;
    private final GolekPluginRegistry kernelPluginRegistry;
    private final ReliabilityManager reliabilityManager;
    private final ExecutorService executorService;

    private volatile boolean initialized = false;
    private volatile boolean started = false;

    public SystemModule() {
        // Initialize core components
        this.kernelModule = new KernelModule();
        this.pluginManager = kernelModule.getPluginManager();
        this.observabilityManager = kernelModule.getObservabilityManager();
        this.providerRegistry = kernelModule.getProviderRegistry();
        this.kernelPluginRegistry = kernelModule.getKernelPluginRegistry();

        // Initialize enhanced configuration manager
        this.configurationManager = new EnhancedConfigurationManager(kernelModule.getConfigurationManager());

        // Initialize reliability manager
        this.reliabilityManager = new ReliabilityManager();

        // Initialize executor service for async operations
        this.executorService = Executors.newCachedThreadPool();
    }

    /**
     * Initialize the system module asynchronously
     */
    public CompletableFuture<Void> initializeAsync() {
        return CompletableFuture.runAsync(() -> {
            try {
                initialize();
            } catch (Exception e) {
                throw new RuntimeException("System module initialization failed", e);
            }
        }, executorService);
    }

    /**
     * Initialize the system module synchronously
     */
    public void initialize() {
        if (initialized) {
            LOG.warn("System module already initialized");
            return;
        }

        LOG.info("Initializing system module");

        try {
            // Initialize kernel module first
            LOG.debug("Initializing kernel module");
            kernelModule.initialize();

            // Initialize configuration manager
            LOG.debug("Initializing configuration manager");

            // Initialize observability manager
            LOG.debug("Initializing observability manager");

            // Initialize reliability manager
            LOG.debug("Initializing reliability manager");

            initialized = true;
            LOG.info("System module initialized successfully");

        } catch (Exception e) {
            LOG.error("Failed to initialize system module", e);
            throw new RuntimeException("System initialization failed", e);
        }
    }

    /**
     * Start the system module asynchronously
     */
    public CompletableFuture<Void> startAsync() {
        return CompletableFuture.runAsync(() -> {
            try {
                start();
            } catch (Exception e) {
                throw new RuntimeException("System module startup failed", e);
            }
        }, executorService);
    }

    /**
     * Start the system module synchronously
     */
    public void start() {
        if (!initialized) {
            throw new IllegalStateException("System module not initialized");
        }

        if (started) {
            LOG.warn("System module already started");
            return;
        }

        LOG.info("Starting system module");

        try {
            // Start kernel module
            LOG.debug("Starting kernel module");
            kernelModule.start();

            started = true;
            LOG.info("System module started successfully");

        } catch (Exception e) {
            LOG.error("Failed to start system module", e);
            throw new RuntimeException("System startup failed", e);
        }
    }

    /**
     * Stop the system module asynchronously
     */
    public CompletableFuture<Void> stopAsync() {
        return CompletableFuture.runAsync(() -> {
            try {
                stop();
            } catch (Exception e) {
                throw new RuntimeException("System module shutdown failed", e);
            }
        }, executorService);
    }

    /**
     * Stop the system module synchronously
     */
    public void stop() {
        if (!started) {
            LOG.warn("System module not started");
            return;
        }

        LOG.info("Stopping system module");

        try {
            // Stop kernel module
            LOG.debug("Stopping kernel module");
            kernelModule.stop();

            // Shutdown executor service
            LOG.debug("Shutting down executor service");
            executorService.shutdown();

            started = false;
            LOG.info("System module stopped successfully");

        } catch (Exception e) {
            LOG.error("Failed to stop system module", e);
            throw new RuntimeException("System shutdown failed", e);
        }
    }

    /**
     * Get the kernel module
     */
    @Produces
    @Singleton
    public KernelModule getKernelModule() {
        return kernelModule;
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
     * Get the enhanced configuration manager
     */
    @Produces
    @Singleton
    public EnhancedConfigurationManager getEnhancedConfigurationManager() {
        return configurationManager;
    }

    /**
     * Get the observability manager
     */
    @Produces
    @Singleton
    public ObservabilityManager getObservabilityManager() {
        return observabilityManager;
    }

    /**
     * Get the enhanced provider registry
     */
    @Produces
    @Singleton
    public EnhancedProviderRegistry getEnhancedProviderRegistry() {
        return providerRegistry;
    }

    /**
     * Get the kernel plugin registry
     */
    @Produces
    @Singleton
    public GolekPluginRegistry getKernelPluginRegistry() {
        return kernelPluginRegistry;
    }

    /**
     * Get the reliability manager
     */
    @Produces
    @Singleton
    public ReliabilityManager getReliabilityManager() {
        return reliabilityManager;
    }

    /**
     * Get the executor service
     */
    @Produces
    @Singleton
    public ExecutorService getExecutorService() {
        return executorService;
    }

    /**
     * Check if system is initialized
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Check if system is started
     */
    public boolean isStarted() {
        return started;
    }

    /**
     * Get system module status
     */
    public SystemStatus getStatus() {
        return new SystemStatus(initialized, started);
    }

    /**
     * System status record
     */
    public record SystemStatus(boolean initialized, boolean started) {
    }
}