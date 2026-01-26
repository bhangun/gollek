package tech.kayys.golek.core.plugin;

/**
 * Extension point interface
 */
public interface ExtensionPoint {
    /**
     * Get extension point ID
     */
    String getId();
    
    /**
     * Get extension point name
     */
    String getName();
    
    /**
     * Get extension point type
     */
    Class<?> getExtensionType();
    
    /**
     * Get extension point description
     */
    String getDescription();
}