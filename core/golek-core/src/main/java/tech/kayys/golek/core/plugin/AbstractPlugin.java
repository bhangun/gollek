package tech.kayys.golek.core.plugin;

import tech.kayys.golek.spi.plugin.GolekPlugin;
import tech.kayys.golek.spi.plugin.PluginContext;

/**
 * Abstract base plugin implementation
 */
public abstract class AbstractPlugin implements GolekPlugin {
    private final String id;
    private final String version;
    private final PluginMetadata metadata;

    protected AbstractPlugin(String id, String version) {
        this.id = id;
        this.version = version;
        this.metadata = new PluginMetadata(id, version, getClass().getSimpleName(), 100);
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public String version() {
        return version;
    }

    @Override
    public PluginMetadata metadata() {
        return metadata;
    }

    @Override
    public void initialize(PluginContext context) {
        onInitialize(context);
    }

    @Override
    public void start() {
        onStart();
    }

    @Override
    public void stop() {
        onStop();
    }

    @Override
    public void shutdown() {
        onDestroy();
    }

    /**
     * Called during plugin initialization
     */
    protected void onInitialize(PluginContext context) {
        // Subclasses can override
    }

    /**
     * Called when plugin starts
     */
    protected void onStart() {
        // Subclasses can override
    }

    /**
     * Called when plugin stops
     */
    protected void onStop() {
        // Subclasses can override
    }

    /**
     * Called when plugin is destroyed
     */
    protected void onDestroy() {
        // Subclasses can override
    }
}