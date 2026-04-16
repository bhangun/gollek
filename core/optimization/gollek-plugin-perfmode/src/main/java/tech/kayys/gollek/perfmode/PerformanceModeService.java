package tech.kayys.gollek.perfmode;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.jboss.logging.Logger;
import tech.kayys.gollek.engine.inference.DefaultInferenceOrchestrator;
import tech.kayys.gollek.spi.batch.BatchConfig;
import tech.kayys.gollek.spi.batch.BatchScheduler;
import tech.kayys.gollek.spi.batch.BatchStrategy;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Performance Mode Service — switches Gollek's live batch and disaggregation
 * settings between three tuned profiles at runtime without a restart.
 *
 * <p>Profiles adjust {@link DefaultInferenceOrchestrator} disaggregation mode
 * and small-prompt threshold, and reconfigure the {@link BatchScheduler} via
 * {@link BatchScheduler#setConfig(BatchConfig)}.
 *
 * <h2>Profiles</h2>
 * <ul>
 *   <li><b>BALANCED</b> — DYNAMIC batching (50 ms window), disaggregation off,
 *       small-prompt threshold 128 tokens.</li>
 *   <li><b>INTERACTIVITY</b> — DYNAMIC batching (5 ms window), disaggregation off,
 *       threshold 32 tokens. Minimises TTFT for interactive chat.</li>
 *   <li><b>THROUGHPUT</b> — CONTINUOUS batching, disaggregation on,
 *       threshold 512 tokens. Maximises tokens/second for batch pipelines.</li>
 * </ul>
 *
 * <h2>REST API</h2>
 * <pre>
 *   GET  /v1/perf-mode                    → current global profile + tenant override count
 *   PUT  /v1/perf-mode/{profile}          → set global profile
 *   GET  /v1/perf-mode/tenant/{tenantId}  → get effective profile for tenant
 *   PUT  /v1/perf-mode/tenant/{tenantId}/{profile} → set tenant override
 *   DELETE /v1/perf-mode/tenant/{tenantId}  → clear tenant override
 * </pre>
 */
@ApplicationScoped
@Path("/v1/perf-mode")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class PerformanceModeService {

    private static final Logger LOG = Logger.getLogger(PerformanceModeService.class);

    public enum Profile { BALANCED, INTERACTIVITY, THROUGHPUT }

    @Inject DefaultInferenceOrchestrator orchestrator;
    @Inject BatchScheduler batchScheduler;

    private volatile Profile globalProfile = Profile.BALANCED;
    private final Map<String, Profile> tenantOverrides = new ConcurrentHashMap<>();

    // ── Global ────────────────────────────────────────────────────────────────

    @GET
    public Response getGlobal() {
        return Response.ok(Map.of(
                "profile",          globalProfile.name(),
                "tenant_overrides", tenantOverrides.size()
        )).build();
    }

    @PUT @Path("/{profile}")
    public Response setGlobal(@PathParam("profile") String name) {
        Profile p = parse(name);
        if (p == null) return Response.status(400)
                .entity(Map.of("error", "unknown profile: " + name)).build();
        Profile prev = globalProfile;
        applyProfile(p);
        globalProfile = p;
        LOG.infof("[PerfMode] Global: %s → %s", prev, p);
        return Response.ok(Map.of("profile", p.name(), "previous", prev.name())).build();
    }

    // ── Per-tenant overrides ──────────────────────────────────────────────────

    @GET @Path("/tenant/{tenantId}")
    public Response getTenant(@PathParam("tenantId") String tenantId) {
        Profile ov  = tenantOverrides.get(tenantId);
        Profile eff = ov != null ? ov : globalProfile;
        return Response.ok(Map.of(
                "tenant_id", tenantId,
                "override",  ov != null ? ov.name() : null,
                "effective", eff.name()
        )).build();
    }

    @PUT @Path("/tenant/{tenantId}/{profile}")
    public Response setTenant(@PathParam("tenantId") String tenantId,
                               @PathParam("profile")  String name) {
        Profile p = parse(name);
        if (p == null) return Response.status(400)
                .entity(Map.of("error", "unknown profile: " + name)).build();
        tenantOverrides.put(tenantId, p);
        LOG.infof("[PerfMode] Tenant %s override → %s", tenantId, p);
        return Response.ok(Map.of("tenant_id", tenantId, "profile", p.name())).build();
    }

    @DELETE @Path("/tenant/{tenantId}")
    public Response clearTenant(@PathParam("tenantId") String tenantId) {
        Profile removed = tenantOverrides.remove(tenantId);
        return Response.ok(Map.of(
                "tenant_id", tenantId,
                "cleared",   removed != null,
                "effective", globalProfile.name()
        )).build();
    }

    // ── Programmatic API ──────────────────────────────────────────────────────

    public Profile effectiveProfile(String tenantId) {
        return tenantOverrides.getOrDefault(tenantId, globalProfile);
    }

    // ── Profile application ───────────────────────────────────────────────────

    private void applyProfile(Profile p) {
        switch (p) {
            case INTERACTIVITY -> {
                orchestrator.setDisaggregatedMode(false);
                orchestrator.setSmallPromptThreshold(32);
                batchScheduler.setConfig(new BatchConfig(
                        BatchStrategy.DYNAMIC, 8, Duration.ofMillis(5), 4, 4, 32, false));
            }
            case THROUGHPUT -> {
                orchestrator.setDisaggregatedMode(true);
                orchestrator.setSmallPromptThreshold(512);
                batchScheduler.setConfig(new BatchConfig(
                        BatchStrategy.CONTINUOUS, 32, Duration.ofMillis(200), 8, 8, 512, true));
            }
            default -> { // BALANCED
                orchestrator.setDisaggregatedMode(false);
                orchestrator.setSmallPromptThreshold(128);
                batchScheduler.setConfig(BatchConfig.defaultDynamic());
            }
        }
        LOG.infof("[PerfMode] Applied profile %s to orchestrator + scheduler", p);
    }

    private Profile parse(String name) {
        try { return Profile.valueOf(name.toUpperCase()); }
        catch (IllegalArgumentException e) { return null; }
    }
}
