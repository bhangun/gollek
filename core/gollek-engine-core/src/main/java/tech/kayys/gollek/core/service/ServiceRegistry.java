package tech.kayys.gollek.core.service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service registry for managing services in the system
 */
public class ServiceRegistry {
    private final Map<Class<?>, Object> services = new ConcurrentHashMap<>();
    private final Map<String, Object> namedServices = new ConcurrentHashMap<>();
    private final Map<Class<?>, List<Object>> serviceLists = new ConcurrentHashMap<>();
    
    /**
     * Register a service
     */
    public <T> void registerService(Class<T> serviceType, T service) {
        services.put(serviceType, service);
        
        // Also add to list for multiple implementations
        serviceLists.computeIfAbsent(serviceType, k -> new ArrayList<>()).add(service);
    }
    
    /**
     * Register a named service
     */
    public void registerNamedService(String name, Object service) {
        namedServices.put(name, service);
    }
    
    /**
     * Get service by type
     */
    @SuppressWarnings("unchecked")
    public <T> T getService(Class<T> serviceType) {
        return (T) services.get(serviceType);
    }
    
    /**
     * Get named service
     */
    public Object getNamedService(String name) {
        return namedServices.get(name);
    }
    
    /**
     * Get all services of a specific type
     */
    @SuppressWarnings("unchecked")
    public <T> List<T> getServices(Class<T> serviceType) {
        List<Object> list = serviceLists.get(serviceType);
        if (list == null) {
            return Collections.emptyList();
        }
        
        List<T> result = new ArrayList<>();
        for (Object obj : list) {
            if (serviceType.isInstance(obj)) {
                result.add((T) obj);
            }
        }
        return Collections.unmodifiableList(result);
    }
    
    /**
     * Check if service exists
     */
    public boolean hasService(Class<?> serviceType) {
        return services.containsKey(serviceType);
    }
    
    /**
     * Check if named service exists
     */
    public boolean hasNamedService(String name) {
        return namedServices.containsKey(name);
    }
    
    /**
     * Unregister service by type
     */
    public <T> void unregisterService(Class<T> serviceType) {
        services.remove(serviceType);
        serviceLists.remove(serviceType);
    }
    
    /**
     * Unregister named service
     */
    public void unregisterNamedService(String name) {
        namedServices.remove(name);
    }
    
    /**
     * Get all registered service types
     */
    public Set<Class<?>> getServiceTypes() {
        return Collections.unmodifiableSet(services.keySet());
    }
    
    /**
     * Get all registered service names
     */
    public Set<String> getServiceNames() {
        return Collections.unmodifiableSet(namedServices.keySet());
    }
    
    /**
     * Clear all services
     */
    public void clear() {
        services.clear();
        namedServices.clear();
        serviceLists.clear();
    }
}