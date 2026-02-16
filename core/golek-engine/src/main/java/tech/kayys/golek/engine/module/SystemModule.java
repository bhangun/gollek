package tech.kayys.golek.engine.module;

import tech.kayys.golek.core.config.EnhancedConfigurationManager;
import tech.kayys.golek.core.plugin.PluginManager;
import tech.kayys.golek.engine.plugin.GolekPluginRegistry;
import tech.kayys.golek.engine.model.ReliabilityManager;
import tech.kayys.golek.spi.provider.ProviderRegistry;
import tech.kayys.golek.engine.model.CachedModelRepository;
import tech.kayys.golek.engine.model.ModelRunnerFactory;

import org.jboss.logging.Logger;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import java.util.Map;
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
    private final ProviderRegistry providerRegistry;
    private final GolekPluginRegistry kernelPluginRegistry;
    private final ReliabilityManager reliabilityManager;
    private final ExecutorService executorService;
    private final ModelRunnerFactory runnerFactory;
    private final CachedModelRepository modelRepository;

    private volatile boolean initialized = false;
    private volatile boolean started = false;

    public interface RequestConfigRepository {
        Map<String, Object> getRunnerConfig(String requestId, String runnerId);

        boolean isQuotaExhausted(String requestId, String providerId);

        boolean isCostSensitive(String requestId);

        void updateConfig(String requestId, RequestConfig config);
    }

    public record RequestConfig(Map<String, Object> properties) {
    }

    @Inject
    public SystemModule(ProviderRegistry providerRegistry,
            ModelRunnerFactory runnerFactory,
            CachedModelRepository modelRepository) {
        this.providerRegistry = providerRegistry;
        this.runnerFactory = runnerFactory;
        this.modelRepository = modelRepository;
        // Initialize core components
        this.kernelModule = new KernelModule(providerRegistry);
        this.pluginManager = kernelModule.getPluginManager();
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

            // Register local runner bridges
            registerLocalRunners();

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

    private void registerLocalRunners() {
        try {
            for (String runnerName : runnerFactory.getAvailableRunners()) {
                LOG.infof("Registering bridge provider for local runner: %s", runnerName);
                var provider = new tech.kayys.golek.engine.provider.adapter.RunnerBridgeProvider(
                        runnerName, runnerFactory, modelRepository);
                providerRegistry.register(provider);
            }
        } catch (Exception e) {
            LOG.warn("Failed to register local runners", e);
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
     * Get the kernel plugin registry
     */
    public GolekPluginRegistry getKernelPluginRegistry() {
        return kernelPluginRegistry;
    }

    /**
     * Get the reliability manager
     */
    public ReliabilityManager getReliabilityManager() {
        return reliabilityManager;
    }

    /**
     * Produce HardwareDetector bean
     */
    @Produces
    @Singleton
    public tech.kayys.golek.model.core.HardwareDetector produceHardwareDetector() {
        return new tech.kayys.golek.model.core.HardwareDetector();
    }

    @Produces
    @Singleton
    public RequestConfigRepository produceTenantConfigRepository() {
        return new RequestConfigRepository() {
            @Override
            public Map<String, Object> getRunnerConfig(String requestId, String runnerId) {
                return Map.of();
            }

            @Override
            public boolean isQuotaExhausted(String requestId, String providerId) {
                return false;
            }

            @Override
            public boolean isCostSensitive(String requestId) {
                return false;
            }

            @Override
            public void updateConfig(String requestId, RequestConfig config) {
                // No-op
            }
        };
    }

    /**
     * Produce RunnerMetrics bean
     */
    @Produces
    @Singleton
    public tech.kayys.golek.model.core.RunnerMetrics produceRunnerMetrics() {
        return new tech.kayys.golek.model.core.RunnerMetrics() {
            @Override
            public java.util.Optional<java.time.Duration> getP95Latency(String runnerName, String modelId) {
                return java.util.Optional.empty();
            }

            @Override
            public boolean isHealthy(String runnerName) {
                return true;
            }

            @Override
            public double getCurrentLoad(String runnerName) {
                return 0.0;
            }
        };
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
