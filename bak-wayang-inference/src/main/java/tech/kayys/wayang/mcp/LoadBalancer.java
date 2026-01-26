package tech.kayys.golek.mcp;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class LoadBalancer {
    
    private final Map<String, Backend> backends = new ConcurrentHashMap<>();
    private final AtomicInteger roundRobinCounter = new AtomicInteger(0);
    private final LoadBalancingStrategy strategy;
    
    public LoadBalancer(LoadBalancingStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void registerBackend(String id, String endpoint, int weight) {
        backends.put(id, new Backend(id, endpoint, weight, 0, true));
    }
    
    public void unregisterBackend(String id) {
        backends.remove(id);
    }
    
    public Backend selectBackend() {
        if (backends.isEmpty()) {
            return null;
        }
        
        List<Backend> available = backends.values().stream()
            .filter(Backend::healthy)
            .toList();
        
        if (available.isEmpty()) {
            return null;
        }
        
        return switch (strategy) {
            case ROUND_ROBIN -> selectRoundRobin(available);
            case LEAST_CONNECTIONS -> selectLeastConnections(available);
            case WEIGHTED -> selectWeighted(available);
            case RANDOM -> selectRandom(available);
        };
    }
    
    private Backend selectRoundRobin(List<Backend> available) {
        int index = roundRobinCounter.getAndIncrement() % available.size();
        return available.get(index);
    }
    
    private Backend selectLeastConnections(List<Backend> available) {
        return available.stream()
            .min(Comparator.comparingInt(Backend::activeConnections))
            .orElse(null);
    }
    
    private Backend selectWeighted(List<Backend> available) {
        int totalWeight = available.stream()
            .mapToInt(Backend::weight)
            .sum();
        
        int random = new Random().nextInt(totalWeight);
        int cumulative = 0;
        
        for (Backend backend : available) {
            cumulative += backend.weight();
            if (random < cumulative) {
                return backend;
            }
        }
        
        return available.get(0);
    }
    
    private Backend selectRandom(List<Backend> available) {
        int index = new Random().nextInt(available.size());
        return available.get(index);
    }
    
    public void incrementConnections(String backendId) {
        Backend backend = backends.get(backendId);
        if (backend != null) {
            backend.incrementConnections();
        }
    }
    
    public void decrementConnections(String backendId) {
        Backend backend = backends.get(backendId);
        if (backend != null) {
            backend.decrementConnections();
        }
    }
    
    public void markUnhealthy(String backendId) {
        Backend backend = backends.get(backendId);
        if (backend != null) {
            backend.setHealthy(false);
        }
    }
    
    public enum LoadBalancingStrategy {
        ROUND_ROBIN,
        LEAST_CONNECTIONS,
        WEIGHTED,
        RANDOM
    }
    
    public static class Backend {
        private final String id;
        private final String endpoint;
        private final int weight;
        private final AtomicInteger activeConnections;
        private volatile boolean healthy;
        
        Backend(String id, String endpoint, int weight, int activeConnections, boolean healthy) {
            this.id = id;
            this.endpoint = endpoint;
            this.weight = weight;
            this.activeConnections = new AtomicInteger(activeConnections);
            this.healthy = healthy;
        }
        
        public String id() { return id; }
        public String endpoint() { return endpoint; }
        public int weight() { return weight; }
        public int activeConnections() { return activeConnections.get(); }
        public boolean healthy() { return healthy; }
        
        void incrementConnections() {
            activeConnections.incrementAndGet();
        }
        
        void decrementConnections() {
            activeConnections.decrementAndGet();
        }
        
        void setHealthy(boolean healthy) {
            this.healthy = healthy;
        }
    }
}
