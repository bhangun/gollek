package tech.kayys.wayang.memory.scheduler;

import tech.kayys.wayang.memory.service.MemoryOptimizationService;
import tech.kayys.wayang.memory.service.MemorySecurityService;
import tech.kayys.wayang.memory.service.MemoryService;
import tech.kayys.wayang.memory.entity.MemorySessionEntity;
import io.quarkus.scheduler.Scheduled;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;

@ApplicationScoped
public class MemoryMaintenanceScheduler {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryMaintenanceScheduler.class);
    
    @Inject
    MemoryOptimizationService optimizationService;
    
    @Inject
    MemorySecurityService securityService;
    
    @Inject
    MemoryService memoryService;

    @Scheduled(cron = "0 0 2 * * ?") // Run at 2 AM daily
    public void optimizeStaleMemories() {
        LOG.info("Starting scheduled memory optimization");
        
        MemorySessionEntity.<MemorySessionEntity>findAll().list()
            .onItem().transformToMulti(sessions -> Multi.createFrom().iterable(sessions))
            .onItem().transformToUniAndMerge(session -> 
                optimizationService.optimizeMemory(session.sessionId)
                    .onItem().invoke(result -> 
                        LOG.info("Optimized session: {}, saved: {} bytes", 
                            session.sessionId, result.getSpaceSaved()))
                    .onFailure().invoke(throwable -> 
                        LOG.error("Failed to optimize session: {}", session.sessionId, throwable)))
            .collect().asList()
            .subscribe().with(
                results -> LOG.info("Completed optimization for {} sessions", results.size()),
                throwable -> LOG.error("Optimization job failed", throwable)
            );
    }

    @Scheduled(cron = "0 0 3 * * ?") // Run at 3 AM daily
    public void cleanupExpiredSessions() {
        LOG.info("Starting cleanup of expired sessions");
        
        Instant now = Instant.now();
        
        MemorySessionEntity.<MemorySessionEntity>delete(
                "expiresAt < ?1 AND expiresAt IS NOT NULL", now)
            .subscribe().with(
                count -> LOG.info("Deleted {} expired sessions", count),
                throwable -> LOG.error("Cleanup job failed", throwable)
            );
    }

    @Scheduled(every = "6h") // Run every 6 hours
    public void auditMemorySecurity() {
        LOG.info("Starting security audit of memory");
        
        MemorySessionEntity.<MemorySessionEntity>findAll().list()
            .onItem().transformToMulti(sessions -> Multi.createFrom().iterable(sessions))
            .select().first(100) // Limit to 100 sessions per run
            .onItem().transformToUniAndMerge(session -> 
                memoryService.getContext(session.sessionId, session.userId)
                    .onItem().transformToUni(context -> 
                        securityService.scanMemoryForPII(context))
                    .onItem().invoke(scanResult -> {
                        if (!scanResult.isPassed()) {
                            LOG.warn("Security violations found in session: {}, violations: {}", 
                                session.sessionId, scanResult.getViolations().size());
                        }
                    })
                    .onFailure().invoke(throwable -> 
                        LOG.error("Failed to scan session: {}", session.sessionId, throwable)))
            .collect().asList()
            .subscribe().with(
                results -> LOG.info("Completed security audit for {} sessions", results.size()),
                throwable -> LOG.error("Security audit failed", throwable)
            );
    }
}