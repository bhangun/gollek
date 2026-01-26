package tech.kayys.golek.mcp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import tech.kayys.golek.engine.EnginePlugin;
import tech.kayys.golek.plugin.PluginContext;
import tech.kayys.golek.plugin.PluginException;

public class PromptTemplatePlugin implements EnginePlugin {
    private PluginContext context;
    private final Map<String, Template> templates = new ConcurrentHashMap<>();
    private static final Pattern VARIABLE_PATTERN = Pattern.compile("\\{\\{(\\w+)\\}\\}");
    
    @Override
    public String getId() {
        return "prompt-templates";
    }
    
    @Override
    public String getName() {
        return "Prompt Template Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Reusable prompt templates with variables";
    }
    
    @Override
    public void initialize(PluginContext context) throws PluginException {
        this.context = context;
        
        // Register built-in templates
        registerTemplate("summarize", new Template(
            "summarize",
            "Summarize the following text in {{max_words}} words:\n\n{{text}}",
            List.of("text", "max_words")
        ));
        
        registerTemplate("translate", new Template(
            "translate",
            "Translate the following text from {{source_lang}} to {{target_lang}}:\n\n{{text}}",
            List.of("text", "source_lang", "target_lang")
        ));
        
        registerTemplate("code_review", new Template(
            "code_review",
            "Review this {{language}} code and provide feedback:\n\n```{{language}}\n{{code}}\n```\n\nFocus on: {{focus_areas}}",
            List.of("language", "code", "focus_areas")
        ));
    }
    
    @Override
    public void start() {}
    
    @Override
    public void stop() {
        templates.clear();
    }
    
    public void registerTemplate(String name, Template template) {
        templates.put(name, template);
    }
    
    public String renderTemplate(String templateName, Map<String, String> variables) 
            throws PluginException {
        Template template = templates.get(templateName);
        if (template == null) {
            throw new PluginException("Template not found: " + templateName);
        }
        
        // Validate required variables
        for (String required : template.requiredVars()) {
            if (!variables.containsKey(required)) {
                throw new PluginException("Missing required variable: " + required);
            }
        }
        
        // Replace variables
        String result = template.content();
        Matcher matcher = VARIABLE_PATTERN.matcher(result);
        StringBuffer sb = new StringBuffer();
        
        while (matcher.find()) {
            String varName = matcher.group(1);
            String value = variables.getOrDefault(varName, "");
            matcher.appendReplacement(sb, Matcher.quoteReplacement(value));
        }
        matcher.appendTail(sb);
        
        return sb.toString();
    }
    
    @Override
    public String preprocessPrompt(String prompt) {
        // Check if prompt uses template syntax
        if (prompt.startsWith("@template:")) {
            try {
                String[] parts = prompt.substring(10).split("\\|", 2);
                String templateName = parts[0].trim();
                
                Map<String, String> vars = new HashMap<>();
                if (parts.length > 1) {
                    String[] kvPairs = parts[1].split(",");
                    for (String pair : kvPairs) {
                        String[] kv = pair.split("=", 2);
                        if (kv.length == 2) {
                            vars.put(kv[0].trim(), kv[1].trim());
                        }
                    }
                }
                
                return renderTemplate(templateName, vars);
            } catch (Exception e) {
                return prompt;
            }
        }
        
        return prompt;
    }
    
    public List<Template> listTemplates() {
        return new ArrayList<>(templates.values());
    }
    
    public record Template(
        String name,
        String content,
        List<String> requiredVars
    ) {}
}
