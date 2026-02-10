package tech.kayys.wayang.memory.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.util.concurrent.TimeUnit;

@ApplicationScoped
public class MemoryMetrics {
    
    private final Counter memoryRetrievals;
    private final Counter memoryStores;
    private final Counter cacheHits;
    private final Counter cacheMisses;
    private final Timer contextRetrievalTime;
    private final Timer embeddingGenerationTime;
    private final Counter securityViolations;
    private final Counter optimizationRuns;

    @Inject
    public MemoryMetrics(MeterRegistry registry) {
        this.memoryRetrievals = Counter.builder("memory.retrievals")
            .description("Number of memory context retrievals")
            .tag("service", "memory")
            .register(registry);
        
        this.memoryStores = Counter.builder("memory.stores")
            .description("Number of memory context stores")
            .tag("service", "memory")
            .register(registry);
        
        this.cacheHits = Counter.builder("memory.cache.hits")
            .description("Number of cache hits")
            .tag("service", "memory")
            .register(registry);
        
        this.cacheMisses = Counter.builder("memory.cache.misses")
            .description("Number of cache misses")
            .tag("service", "memory")
            .register(registry);
        
        this.contextRetrievalTime = Timer.builder("memory.context.retrieval.time")
            .description("Time taken to retrieve memory context")
            .tag("service", "memory")
            .register(registry);
        
        this.embeddingGenerationTime = Timer.builder("memory.embedding.generation.time")
            .description("Time taken to generate embeddings")
            .tag("service", "memory")
            .register(registry);
        
        this.securityViolations = Counter.builder("memory.security.violations")
            .description("Number of security violations detected")
            .tag("service", "memory")
            .register(registry);
        
        this.optimizationRuns = Counter.builder("memory.optimization.runs")
            .description("Number of memory optimization runs")
            .tag("service", "memory")
            .register(registry);
    }

    public void recordRetrieval() {
        memoryRetrievals.increment();
    }

    public void recordStore() {
        memoryStores.increment();
    }

    public void recordCacheHit() {
        cacheHits.increment();
    }

    public void recordCacheMiss() {
        cacheMisses.increment();
    }

    public void recordRetrievalTime(long milliseconds) {
        contextRetrievalTime.record(milliseconds, TimeUnit.MILLISECONDS);
    }

    public void recordEmbeddingTime(long milliseconds) {
        embeddingGenerationTime.record(milliseconds, TimeUnit.MILLISECONDS);
    }

    public void recordSecurityViolation() {
        securityViolations.increment();
    }

    public void recordOptimization() {
        optimizationRuns.increment();
    }
}