package com.llamajava.plugins.rag;


public class MyPlugin implements EnginePlugin {
    private PluginContext context;
    
    @Override
    public String getId() {
        return "my-plugin";
    }
    
    @Override
    public String getName() {
        return "My Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "My custom plugin";
    }
    
    @Override
    public void initialize(PluginContext context) throws PluginException {
        this.context = context;
        // Initialize your plugin
    }
    
    @Override
    public void start() throws PluginException {
        // Start your plugin
    }
    
    @Override
    public void stop() throws PluginException {
        // Cleanup
    }
}
