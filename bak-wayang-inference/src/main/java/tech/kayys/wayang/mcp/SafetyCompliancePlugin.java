package tech.kayys.golek.mcp;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import tech.kayys.golek.engine.EnginePlugin;
import tech.kayys.golek.model.GenerationResult;
import tech.kayys.golek.plugin.PluginContext;
import tech.kayys.golek.plugin.PluginException;

public class SafetyCompliancePlugin implements EnginePlugin {
    private PluginContext context;
    private final List<SafetyRule> rules = new ArrayList<>();
    private final List<String> blockedPatterns = new ArrayList<>();
    private boolean strictMode = false;
    
    @Override
    public String getId() {
        return "safety-compliance";
    }
    
    @Override
    public String getName() {
        return "Safety & Compliance Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Advanced content safety and regulatory compliance";
    }
    
    @Override
    public void initialize(PluginContext context) throws PluginException {
        this.context = context;
        
        // Load safety rules
        rules.add(new SafetyRule(
            "pii-detection",
            "Detect and redact PII",
            Pattern.compile("\\b\\d{3}-\\d{2}-\\d{4}\\b"), // SSN pattern
            SafetyAction.REDACT
        ));
        
        rules.add(new SafetyRule(
            "profanity-filter",
            "Block profanity",
            Pattern.compile("\\b(badword1|badword2)\\b", Pattern.CASE_INSENSITIVE),
            SafetyAction.BLOCK
        ));
        
        rules.add(new SafetyRule(
            "medical-disclaimer",
            "Add disclaimer for medical content",
            Pattern.compile("\\b(diagnose|treatment|medication)\\b", Pattern.CASE_INSENSITIVE),
            SafetyAction.WARN
        ));
        
        Boolean configStrictMode = context.getConfigValue("safety.strict_mode", Boolean.class);
        if (configStrictMode != null) {
            strictMode = configStrictMode;
        }
    }
    
    @Override
    public void start() {}
    
    @Override
    public void stop() {
        rules.clear();
    }
    
    @Override
    public String preprocessPrompt(String prompt) {
        for (SafetyRule rule : rules) {
            if (rule.pattern().matcher(prompt).find()) {
                switch (rule.action()) {
                    case BLOCK:
                        if (strictMode) {
                            throw new SecurityException("Content blocked by safety rule: " + rule.name());
                        }
                        break;
                    case REDACT:
                        prompt = rule.pattern().matcher(prompt).replaceAll("[REDACTED]");
                        break;
                    case WARN:
                        context.setSharedData("safety.warning." + rule.name(), true);
                        break;
                }
            }
        }
        
        return prompt;
    }
    
    @Override
    public String postprocessOutput(String output) {
        StringBuilder warnings = new StringBuilder();
        
        for (SafetyRule rule : rules) {
            if (rule.pattern().matcher(output).find()) {
                switch (rule.action()) {
                    case BLOCK:
                        if (strictMode) {
                            return "I cannot provide that information due to safety policies.";
                        }
                        break;
                    case REDACT:
                        output = rule.pattern().matcher(output).replaceAll("[REDACTED]");
                        break;
                    case WARN:
                        if (rule.name().equals("medical-disclaimer")) {
                            warnings.append("\n\n⚠️ Medical Disclaimer: This information is for educational purposes only. Consult a healthcare professional.");
                        }
                        break;
                }
            }
        }
        
        return output + warnings.toString();
    }
    
    @Override
    public void onGenerationComplete(GenerationResult result) {
        // Log safety events
        context.emitEvent("safety.check.complete", Map.of(
            "rules_applied", rules.size(),
            "strict_mode", strictMode
        ));
    }
    
    private record SafetyRule(
        String name,
        String description,
        Pattern pattern,
        SafetyAction action
    ) {}
    
    private enum SafetyAction {
        BLOCK,   // Block the request
        REDACT,  // Redact matching content
        WARN     // Add warning message
    }
}
