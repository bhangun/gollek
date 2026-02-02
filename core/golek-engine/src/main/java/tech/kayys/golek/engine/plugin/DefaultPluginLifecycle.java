package tech.kayys.golek.engine.plugin;

import java.util.concurrent.atomic.AtomicReference;

import tech.kayys.golek.api.plugin.GolekPlugin;
import tech.kayys.golek.api.plugin.PluginContext;
import tech.kayys.golek.api.plugin.PluginState;

/**
 * Default plugin lifecycle implementation.
 */
public class DefaultPluginLifecycle implements PluginLifecycle {

    private final GolekPlugin plugin;
    private final AtomicReference<PluginState> state = new AtomicReference<>(PluginState.CREATED);

    public DefaultPluginLifecycle(GolekPlugin plugin) {
        this.plugin = plugin;
    }

    @Override
    public PluginState getState() {
        return state.get();
    }

    @Override
    public void initialize(PluginContext context) {
        if (state.compareAndSet(PluginState.CREATED, PluginState.INITIALIZING)) {
            try {
                // Initialization logic
                state.set(PluginState.INITIALIZED);
            } catch (Exception e) {
                state.set(PluginState.FAILED);
                throw new RuntimeException("Plugin initialization failed", e);
            }
        }
    }

    @Override
    public void start() {
        if (state.compareAndSet(PluginState.INITIALIZED, PluginState.STARTING)) {
            try {
                // Start logic
                state.set(PluginState.RUNNING);
            } catch (Exception e) {
                state.set(PluginState.FAILED);
                throw new RuntimeException("Plugin start failed", e);
            }
        }
    }

    @Override
    public void stop() {
        if (state.compareAndSet(PluginState.RUNNING, PluginState.STOPPING)) {
            try {
                // Stop logic
                state.set(PluginState.STOPPED);
            } catch (Exception e) {
                state.set(PluginState.FAILED);
                throw new RuntimeException("Plugin stop failed", e);
            }
        }
    }

    @Override
    public void destroy() {
        state.set(PluginState.DESTROYED);
    }

    @Override
    public boolean isInitialized() {
        PluginState currentState = state.get();
        return currentState == PluginState.INITIALIZED ||
                currentState == PluginState.STARTING ||
                currentState == PluginState.RUNNING ||
                currentState == PluginState.STOPPING ||
                currentState == PluginState.STOPPED;
    }

    @Override
    public boolean isRunning() {
        return state.get() == PluginState.RUNNING;
    }
}
