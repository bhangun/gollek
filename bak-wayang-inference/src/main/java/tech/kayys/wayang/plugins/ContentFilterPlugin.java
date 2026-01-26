package tech.kayys.golek.plugins;

import java.util.Set;
import java.util.regex.Pattern;

import tech.kayys.golek.engine.EnginePlugin;
import tech.kayys.golek.plugin.PluginContext;
import tech.kayys.golek.plugin.PluginException;

public class ContentFilterPlugin implements EnginePlugin {
    private PluginContext context;
    private Set<String> blockedWords;
    private Pattern blockedPattern;
    
    @Override
    public String getId() {
        return "content-filter";
    }
    
    @Override
    public String getName() {
        return "Content Filter";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Filters inappropriate content";
    }
    
    @Override
    public void initialize(PluginContext context) throws PluginException {
        this.context = context;
        
        // Load blocked words from config
        String blockedWordsStr = context.getConfigValue("blocked.words", String.class);
        if (blockedWordsStr != null) {
            blockedWords = Set.of(blockedWordsStr.split(","));
            blockedPattern = Pattern.compile(
                String.join("|", blockedWords),
                Pattern.CASE_INSENSITIVE
            );
        }
    }
    
    @Override
    public void start() {}
    
    @Override
    public void stop() {}
    
    @Override
    public String preprocessPrompt(String prompt) {
        if (blockedPattern != null && blockedPattern.matcher(prompt).find()) {
            return "I cannot process that request.";
        }
        return prompt;
    }
    
    @Override
    public String postprocessOutput(String output) {
        if (blockedPattern != null) {
            return blockedPattern.matcher(output).replaceAll("[FILTERED]");
        }
        return output;
    }
}
