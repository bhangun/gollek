package tech.kayys.golek.plugins;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import tech.kayys.golek.engine.EnginePlugin;
import tech.kayys.golek.model.GenerationResult;
import tech.kayys.golek.plugin.PluginContext;

public class AnalyticsPlugin implements EnginePlugin {
    private PluginContext context;
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalTokens = new AtomicLong(0);
    private final AtomicLong totalTime = new AtomicLong(0);
    private final AtomicInteger activeRequests = new AtomicInteger(0);
    
    @Override
    public String getId() {
        return "analytics";
    }
    
    @Override
    public String getName() {
        return "Analytics Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Tracks usage analytics";
    }
    
    @Override
    public void initialize(PluginContext context) {
        this.context = context;
        
        // Listen to events
        context.addEventListener("generation.start", event -> {
            activeRequests.incrementAndGet();
        });
        
        context.addEventListener("generation.complete", event -> {
            activeRequests.decrementAndGet();
        });
    }
    
    @Override
    public void start() {}
    
    @Override
    public void stop() {
        printStats();
    }
    
    @Override
    public void onGenerationStart(String prompt) {
        totalRequests.incrementAndGet();
    }
    
    @Override
    public void onGenerationComplete(GenerationResult result) {
        totalTokens.addAndGet(result.tokensGenerated());
        totalTime.addAndGet(result.timeMs());
    }
    
    public void printStats() {
        long requests = totalRequests.get();
        System.out.println("=== Analytics ===");
        System.out.println("Total requests: " + requests);
        System.out.println("Total tokens: " + totalTokens.get());
        System.out.println("Average tokens/request: " + (requests > 0 ? totalTokens.get() / requests : 0));
        System.out.println("Average time: " + (requests > 0 ? totalTime.get() / requests : 0) + "ms");
        System.out.println("Active requests: " + activeRequests.get());
    }
}
