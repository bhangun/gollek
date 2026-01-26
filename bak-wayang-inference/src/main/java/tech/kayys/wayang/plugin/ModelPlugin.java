package tech.kayys.golek.plugin;

import tech.kayys.golek.model.ModelInfo;

public interface ModelPlugin extends Plugin {
    
    /**
     * Check if this plugin can handle the model
     */
    boolean canHandle(String modelPath);
    
    /**
     * Load and prepare model
     */
    void prepareModel(String modelPath) throws PluginException;
    
    /**
     * Get model information
     */
    ModelInfo getModelInfo();
    
    /**
     * Called before model is loaded
     */
    default void onModelLoad(String modelPath) {}
    
    /**
     * Called after model is loaded
     */
    default void onModelLoaded(ModelInfo info) {}
    
    /**
     * Called before model is unloaded
     */
    default void onModelUnload() {}
}
