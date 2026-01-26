package tech.kayys.golek.mcp;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.golek.model.ModelInfo;
import tech.kayys.golek.plugin.ModelPlugin;
import tech.kayys.golek.plugin.PluginContext;
import tech.kayys.golek.plugin.PluginException;

public class ModelVersioningPlugin implements ModelPlugin {
    private PluginContext context;
    private final Map<String, ModelVersion> versions = new ConcurrentHashMap<>();
    private String activeVersion;
    
    @Override
    public String getId() {
        return "model-versioning";
    }
    
    @Override
    public String getName() {
        return "Model Versioning Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Version control for models with rollback capability";
    }
    
    @Override
    public void initialize(PluginContext context) throws PluginException {
        this.context = context;
        
        // Register versions
        versions.put("v1.0", new ModelVersion(
            "v1.0",
            "/models/llama-2-7b-v1.gguf",
            "Production stable",
            System.currentTimeMillis(),
            true
        ));
        
        versions.put("v1.1", new ModelVersion(
            "v1.1",
            "/models/llama-2-7b-v1.1.gguf",
            "Improved accuracy",
            System.currentTimeMillis(),
            false
        ));
        
        activeVersion = "v1.0";
    }
    
    @Override
    public void start() {}
    
    @Override
    public void stop() {
        versions.clear();
    }
    
    @Override
    public boolean canHandle(String modelPath) {
        return versions.values().stream()
            .anyMatch(v -> v.path().equals(modelPath));
    }
    
    @Override
    public void prepareModel(String modelPath) throws PluginException {
        // Model preparation logic
    }
    
    @Override
    public ModelInfo getModelInfo() {
        ModelVersion version = versions.get(activeVersion);
        if (version == null) {
            return null;
        }
        
        return ModelInfo.builder()
            .name("llama-2-7b-" + version.version())
            .description(version.description())
            .build();
    }
    
    public void promoteVersion(String versionId) throws PluginException {
        if (!versions.containsKey(versionId)) {
            throw new PluginException("Version not found: " + versionId);
        }
        
        ModelVersion oldVersion = versions.get(activeVersion);
        activeVersion = versionId;
        
        context.emitEvent("model.version.changed", Map.of(
            "old_version", oldVersion.version(),
            "new_version", versionId
        ));
    }
    
    public void rollback() throws PluginException {
        // Rollback to previous stable version
        ModelVersion stable = versions.values().stream()
            .filter(ModelVersion::stable)
            .findFirst()
            .orElseThrow(() -> new PluginException("No stable version found"));
        
        promoteVersion(stable.version());
    }
    
    private record ModelVersion(
        String version,
        String path,
        String description,
        long timestamp,
        boolean stable
    ) {}
}
