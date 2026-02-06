package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Timeout;
import org.jboss.logging.Logger;

import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.engine.plugin.GolekPluginRegistry;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.engine.execution.ExecutionSignal;
import tech.kayys.golek.engine.execution.ExecutionStateMachine;
import tech.kayys.golek.engine.execution.ExecutionStatus;
import tech.kayys.golek.core.inference.InferenceObserver;
import tech.kayys.golek.core.pipeline.InferencePipeline;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Default implementation of the inference pipeline.
 * Executes phases in deterministic order with comprehensive error handling,
 * observability hooks, and state management.
 * 
 * Design principles:
 * - Phases execute in strict order (no branching)
 * - Each phase can have multiple plugins
 * - Plugins execute sequentially within a phase
 * - State transitions are managed via ExecutionStateMachine
 * - All transitions are audited
 * - Circuit breaker protection on critical phases
 */
@ApplicationScoped
public class DefaultInferencePipeline implements InferencePipeline {

        private static final Logger LOG = Logger.getLogger(DefaultInferencePipeline.class);

        @Inject
        GolekPluginRegistry pluginRegistry;

        @Inject
        ExecutionStateMachine stateMachine;

        @Inject
        InferenceObserver observer;

        // Cache of plugins organized by phase for performance
        private volatile Map<InferencePhase, List<InferencePhasePlugin>> pluginCache;

        /**
         * Execute complete pipeline through all phases
         */
        @Override
        @Timeout(value = 30, unit = java.util.concurrent.TimeUnit.SECONDS)
        public Uni<ExecutionContext> execute(ExecutionContext context) {
                LOG.debugf("Starting pipeline execution for request: %s",
                                context.token().getRequestId());

                // Initialize plugin cache if needed
                if (pluginCache == null) {
                        synchronized (this) {
                                if (pluginCache == null) {
                                        pluginCache = buildPluginCache();
                                }
                        }
                }

                // Signal start and update state
                context.updateStatus(
                                stateMachine.next(context.token().getStatus(), ExecutionSignal.START));

                // Notify observers
                observer.onStart(context);

                // Execute phases sequentially
                Uni<ExecutionContext> pipeline = Uni.createFrom().item(context);

                for (InferencePhase phase : InferencePhase.ordered()) {
                        pipeline = pipeline.chain(ctx -> executePhase(ctx, phase));
                }

                return pipeline
                                .onItem().invoke(ctx -> {
                                        // Mark as completed
                                        ctx.updateStatus(ExecutionStatus.COMPLETED);
                                        observer.onSuccess(ctx);
                                        LOG.infof("Pipeline completed successfully for request: %s",
                                                        ctx.token().getRequestId());
                                })
                                .onFailure().invoke(error -> {
                                        LOG.errorf(error, "Pipeline failed for request: %s",
                                                        context.token().getRequestId());
                                        observer.onFailure(error, context);
                                });
        }

        /**
         * Execute a single phase with all its plugins
         */
        @Override
        @CircuitBreaker(requestVolumeThreshold = 10, failureRatio = 0.5, delay = 5000, successThreshold = 2)
        public Uni<ExecutionContext> executePhase(
                        ExecutionContext context,
                        InferencePhase phase) {
                LOG.debugf("Executing phase: %s for request: %s",
                                phase, context.token().getRequestId());

                // Update phase in context
                context.updatePhase(phase);

                // Notify observers
                observer.onPhase(phase, context);

                // Get plugins for this phase
                List<InferencePhasePlugin> plugins = pluginCache.getOrDefault(
                                phase,
                                List.of());

                if (plugins.isEmpty()) {
                        LOG.debugf("No plugins registered for phase: %s", phase);
                        return Uni.createFrom().item(context);
                }

                // Execute plugins sequentially
                Uni<ExecutionContext> chain = Uni.createFrom().item(context);

                for (InferencePhasePlugin plugin : plugins) {
                        chain = chain.chain(ctx -> executePlugin(ctx, plugin, phase));
                }

                return chain
                                .onItem().invoke(ctx -> {
                                        LOG.debugf("Phase %s completed successfully", phase);
                                        // Signal phase success
                                        ctx.updateStatus(
                                                        stateMachine.next(
                                                                        ctx.token().getStatus(),
                                                                        ExecutionSignal.PHASE_SUCCESS));
                                })
                                .onFailure().recoverWithUni(error -> handlePhaseFailure(context, phase, error));
        }

        /**
         * Execute a single plugin with timeout and error handling
         */
        private Uni<ExecutionContext> executePlugin(
                        ExecutionContext context,
                        InferencePhasePlugin plugin,
                        InferencePhase phase) {
                LOG.tracef("Executing plugin: %s in phase: %s",
                                plugin.id(), phase);

                return Uni.createFrom().item(() -> {
                        try {
                                plugin.execute(context, context.engine());
                                return context;
                        } catch (Exception e) {
                                LOG.errorf(e, "Plugin %s failed in phase %s",
                                                plugin.id(), phase);
                                throw new PluginExecutionException(
                                                "Plugin " + plugin.id() + " failed", e);
                        }
                })
                                .runSubscriptionOn(
                                                context.engine().executorService())
                                .ifNoItem().after(Duration.ofSeconds(10))
                                .failWith(() -> new PluginTimeoutException(
                                                "Plugin " + plugin.id() + " timed out"));
        }

        /**
         * Handle phase execution failure with retry logic
         */
        private Uni<ExecutionContext> handlePhaseFailure(
                        ExecutionContext context,
                        InferencePhase phase,
                        Throwable error) {
                LOG.warnf(error, "Phase %s failed for request: %s",
                                phase, context.token().getRequestId());

                context.setError(error);

                // Check if phase is retryable
                if (phase.isRetryable() &&
                                context.token().getAttempt() < getMaxRetries(context)) {

                        LOG.infof("Retrying phase %s (attempt %d)",
                                        phase, context.token().getAttempt() + 1);

                        context.incrementAttempt();
                        context.updateStatus(ExecutionStatus.RETRYING);

                        // Exponential backoff
                        Duration delay = Duration.ofMillis(
                                        100L * (long) Math.pow(2, context.token().getAttempt()));

                        return Uni.createFrom().item(context)
                                        .onItem().delayIt().by(delay)
                                        .chain(ctx -> executePhase(ctx, phase));
                }

                // Mark as failed if retries exhausted
                context.updateStatus(
                                stateMachine.next(
                                                context.token().getStatus(),
                                                ExecutionSignal.RETRY_EXHAUSTED));

                return Uni.createFrom().failure(error);
        }

        /**
         * Build plugin cache organized by phase
         */
        private Map<InferencePhase, List<InferencePhasePlugin>> buildPluginCache() {
                LOG.info("Building plugin cache");

                List<InferencePhasePlugin> allPlugins = pluginRegistry.byType(InferencePhasePlugin.class);

                return allPlugins.stream()
                                .collect(Collectors.groupingBy(
                                                InferencePhasePlugin::phase,
                                                Collectors.collectingAndThen(
                                                                Collectors.toList(),
                                                                list -> list.stream()
                                                                                .sorted((p1, p2) -> Integer.compare(
                                                                                                p1.order(), p2.order()))
                                                                                .toList())));
        }

        /**
         * Get max retries from context metadata or default
         */
        private int getMaxRetries(ExecutionContext context) {
                return context.metadata()
                                .getOrDefault("max.retries", 3)
                                .toString()
                                .transform(Integer::parseInt)
                                .orElse(3);
        }

        /**
         * Invalidate plugin cache (called when plugins are updated)
         */
        public void invalidateCache() {
                LOG.info("Invalidating plugin cache");
                pluginCache = null;
        }

        /**
         * Plugin execution exception
         */
        public static class PluginExecutionException extends RuntimeException {
                public PluginExecutionException(String message, Throwable cause) {
                        super(message, cause);
                }
        }

        /**
         * Plugin timeout exception
         */
        public static class PluginTimeoutException extends RuntimeException {
                public PluginTimeoutException(String message) {
                        super(message);
                }
        }
}