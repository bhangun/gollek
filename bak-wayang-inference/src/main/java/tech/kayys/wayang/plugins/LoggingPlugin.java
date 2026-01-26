package tech.kayys.golek.plugins;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.golek.engine.EnginePlugin;
import tech.kayys.golek.model.GenerationResult;
import tech.kayys.golek.plugin.PluginContext;

public class LoggingPlugin implements EnginePlugin {
    private static final Logger log = LoggerFactory.getLogger(LoggingPlugin.class);
    private PluginContext context;
    
    @Override
    public String getId() {
        return "logging-plugin";
    }
    
    @Override
    public String getName() {
        return "Logging Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Logs all generation requests and responses";
    }
    
    @Override
    public void initialize(PluginContext context) {
        this.context = context;
        log.info("Logging plugin initialized");
    }
    
    @Override
    public void start() {
        log.info("Logging plugin started");
    }
    
    @Override
    public void stop() {
        log.info("Logging plugin stopped");
    }
    
    @Override
    public void onGenerationStart(String prompt) {
        log.info("Generation started. Prompt length: {}", prompt.length());
    }
    
    @Override
    public void onGenerationComplete(GenerationResult result) {
        log.info("Generation complete. Tokens: {}, Time: {}ms",
            result.tokensGenerated(), result.timeMs());
    }
}
