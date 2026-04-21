/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.runner.safetensor;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.quantizer.gptq.GPTQLoader;
import tech.kayys.gollek.quantizer.gptq.GPTQConfig;
import tech.kayys.gollek.quantizer.gptq.GPTQQuantizerService;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.engine.warmup.DirectSafetensorBackend;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Safetensor runner session implementation.
 * 
 * <p>
 * Wraps the existing DirectSafetensorBackend for plugin-based inference.
 * </p>
 * 
 * @since 2.1.0
 */
public class SafetensorRunnerSession implements RunnerSession {

    private static final Logger LOG = Logger.getLogger(SafetensorRunnerSession.class);

    private final String sessionId;
    private final String modelPath;
    private final Map<String, Object> config;
    private final RunnerPlugin runner;
    private final SafetensorProviderConfig providerConfig;
    private final DirectSafetensorBackend backend;
    private final AtomicBoolean active = new AtomicBoolean(true);
    private final ModelInfo modelInfo;
    
    // GPTQ quantization support
    private GPTQLoader gptqLoader;
    private GPTQQuantizerService quantizerService;
    private boolean isQuantizedModel = false;

    /**
     * Create Safetensor runner session.
     * 
     * @param modelPath      Path to model file
     * @param config         Session configuration
     * @param providerConfig Provider configuration
     * @param backend        Safetensor backend
     */
    public SafetensorRunnerSession(
            String modelPath,
            Map<String, Object> config,
            SafetensorProviderConfig providerConfig,
            DirectSafetensorBackend backend) {

        this.sessionId = UUID.randomUUID().toString();
        this.modelPath = modelPath;
        this.config = config;
        this.runner = new SafetensorRunnerPlugin(providerConfig, backend);
        this.providerConfig = providerConfig;
        this.backend = backend;
        this.quantizerService = new GPTQQuantizerService();

        // Detect if this is a quantized model
        this.isQuantizedModel = detectQuantizedModel(modelPath);
        
        // Load model info (and quantized model if applicable)
        this.modelInfo = loadModelInfo(modelPath);

        LOG.infof("Created Safetensor session %s for model: %s (quantized=%b)", 
                sessionId, modelPath, isQuantizedModel);
    }

    @Override
    public RunnerPlugin getRunner() {
        return runner;
    }

    @Override
    public String getSessionId() {
        return sessionId;
    }

    @Override
    public String getModelPath() {
        return modelPath;
    }

    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        if (!active.get()) {
            return Uni.createFrom().failure(new IllegalStateException("Session is closed: " + sessionId));
        }
        return executeInference(request)
                .onFailure().invoke(e -> LOG.errorf("Inference failed for session %s: %s", sessionId, e.getMessage()));
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        if (!active.get()) {
            return Multi.createFrom().failure(new IllegalStateException("Session is closed: " + sessionId));
        }
        return executeStreamingInference(request)
                .onFailure().invoke(e -> LOG.errorf("Streaming failed for session %s: %s", sessionId, e.getMessage()));
    }

    @Override
    public Map<String, Object> getConfig() {
        return config;
    }

    @Override
    public boolean isActive() {
        return active.get();
    }

    @Override
    public void close() {
        if (active.compareAndSet(true, false)) {
            LOG.infof("Closing Safetensor session %s", sessionId);
            releaseModel();
            
            // Clean up GPTQ resources
            if (gptqLoader != null) {
                try {
                    gptqLoader.close();
                    LOG.infof("GPTQ loader closed for session %s", sessionId);
                } catch (Exception e) {
                    LOG.warnf("Error closing GPTQ loader: %s", e.getMessage());
                }
            }
            
            if (quantizerService != null) {
                try {
                    quantizerService.close();
                    LOG.infof("Quantizer service closed for session %s", sessionId);
                } catch (Exception e) {
                    LOG.warnf("Error closing quantizer service: %s", e.getMessage());
                }
            }
        }
    }

    @Override
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Quantized Model Support
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Detects if the model path contains a quantized (GPTQ) model.
     */
    private boolean detectQuantizedModel(String modelPath) {
        try {
            Path path = Path.of(modelPath);
            
            // Check if it's a directory with safetensors files
            if (java.nio.file.Files.isDirectory(path)) {
                // Look for GPTQ signature tensors
                boolean hasQweight = false;
                boolean hasScales = false;
                
                try (var stream = java.nio.file.Files.list(path)) {
                    var safetensorFiles = stream
                            .filter(p -> p.toString().endsWith(".safetensors"))
                            .toList();
                    
                    // Check first few files for GPTQ signatures
                    for (int i = 0; i < Math.min(3, safetensorFiles.size()); i++) {
                        var file = safetensorFiles.get(i);
                        var header = readSafetensorHeader(file);
                        
                        if (header != null) {
                            for (String tensorName : header.keySet()) {
                                if (tensorName.endsWith(".qweight")) {
                                    hasQweight = true;
                                }
                                if (tensorName.endsWith(".scales")) {
                                    hasScales = true;
                                }
                            }
                        }
                        
                        if (hasQweight && hasScales) {
                            return true;
                        }
                    }
                }
            }
            
            return false;
        } catch (Exception e) {
            LOG.debugf("Failed to detect quantized model: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Loads a quantized GPTQ model if detected.
     */
    private void loadQuantizedModel(String modelPath) {
        if (!isQuantizedModel) {
            return;
        }
        
        try {
            Path path = Path.of(modelPath);
            LOG.infof("Loading quantized GPTQ model from: %s", modelPath);
            
            // Auto-detect config and load
            gptqLoader = quantizerService.loadQuantized(path);
            
            LOG.infof("GPTQ model loaded successfully: %d layers", gptqLoader.getLayerCount());
            
        } catch (IOException e) {
            LOG.errorf("Failed to load quantized model: %s", e.getMessage());
            // Continue without quantized model - backend will handle fallback
        }
    }

    /**
     * Gets the GPTQ loader if available.
     */
    public GPTQLoader getGptqLoader() {
        return gptqLoader;
    }

    /**
     * Checks if this session has a quantized model loaded.
     */
    public boolean hasQuantizedModel() {
        return isQuantizedModel && gptqLoader != null;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Reads safetensor header from a file.
     */
    private Map<String, Object> readSafetensorHeader(Path filePath) {
        try {
            // Simplified header reading - in production use proper safetensor parser
            try (var raf = new java.io.RandomAccessFile(filePath.toFile(), "r")) {
                long lengthLE = raf.readLong();
                long headerLength = Long.reverseBytes(lengthLE);
                
                if (headerLength > 0 && headerLength < 100 * 1024 * 1024) {
                    byte[] jsonBytes = new byte[(int) headerLength];
                    raf.readFully(jsonBytes);
                    
                    var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    var root = mapper.readTree(jsonBytes);
                    
                    Map<String, Object> result = new HashMap<>();
                    var fields = root.fields();
                    while (fields.hasNext()) {
                        var field = fields.next();
                        result.put(field.getKey(), field.getValue());
                    }
                    return result;
                }
            }
        } catch (Exception e) {
            LOG.debugf("Failed to read header from %s: %s", filePath, e.getMessage());
        }
        return null;
    }

    private ModelInfo loadModelInfo(String modelPath) {
        java.nio.file.Path path = java.nio.file.Paths.get(modelPath);
        String modelName = path.getFileName().toString();
        String architecture = detectArchitecture(modelPath);
        long parameters = 0;

        // Load quantized model if detected
        loadQuantizedModel(modelPath);

        try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(modelPath, "r")) {
            long lengthLE = raf.readLong();
            long headerLength = Long.reverseBytes(lengthLE);
            if (headerLength > 0 && headerLength < 100 * 1024 * 1024) {
                byte[] jsonBytes = new byte[(int) headerLength];
                raf.readFully(jsonBytes);

                com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                com.fasterxml.jackson.databind.JsonNode root = mapper.readTree(jsonBytes);

                // Calculate parameters from tensor shapes
                java.util.Iterator<Map.Entry<String, com.fasterxml.jackson.databind.JsonNode>> fields = root.fields();
                while (fields.hasNext()) {
                    Map.Entry<String, com.fasterxml.jackson.databind.JsonNode> field = fields.next();
                    if (!"__metadata__".equals(field.getKey())) {
                        com.fasterxml.jackson.databind.JsonNode tensor = field.getValue();
                        if (tensor.has("shape")) {
                            long tensorParams = 1;
                            for (com.fasterxml.jackson.databind.JsonNode dim : tensor.get("shape")) {
                                tensorParams *= dim.asLong();
                            }
                            parameters += tensorParams;
                        }
                    }
                }
            }
        } catch (Exception e) {
            LOG.warnf("Failed to read Safetensor header from %s: %s", modelPath, e.getMessage());
        }

        if (parameters == 0) {
            parameters = estimateParameters(modelPath);
        }

        int maxContext = 4096;
        if (config.containsKey("max_context")) {
            Object mc = config.get("max_context");
            if (mc instanceof Number)
                maxContext = ((Number) mc).intValue();
            else {
                try {
                    maxContext = Integer.parseInt(mc.toString());
                } catch (NumberFormatException ignored) {
                }
            }
        }

        String device = config.getOrDefault("device", "cpu").toString();

        return ModelInfo.builder()
                .name(modelName)
                .architecture(architecture)
                .parameterCount(String.valueOf(parameters))
                .contextLength((long) maxContext)
                .embeddingSize(4096) // Default embedding size
                .format("Safetensor")
                .metadata(Map.of(
                        "model_name", modelName,
                        "backend", providerConfig.backend(),
                        "device", device,
                        "session_id", sessionId))
                .build();
    }

    private String detectArchitecture(String modelPath) {
        String lower = modelPath.toLowerCase();
        if (lower.contains("llama-3"))
            return "llama3";
        if (lower.contains("llama-2"))
            return "llama2";
        if (lower.contains("llama"))
            return "llama";
        if (lower.contains("mistral"))
            return "mistral";
        if (lower.contains("mixtral"))
            return "mixtral";
        if (lower.contains("qwen"))
            return "qwen";
        if (lower.contains("falcon"))
            return "falcon";
        if (lower.contains("gemma"))
            return "gemma";
        if (lower.contains("phi"))
            return "phi";
        if (lower.contains("bert"))
            return "bert";
        return "unknown";
    }

    private long estimateParameters(String modelPath) {
        // Estimate from filename (e.g., "llama-3-8b" → 8B)
        String lower = modelPath.toLowerCase();
        if (lower.contains("70b"))
            return 70_000_000_000L;
        if (lower.contains("8b") || lower.contains("7b"))
            return 8_000_000_000L;
        if (lower.contains("3b"))
            return 3_000_000_000L;
        if (lower.contains("1b"))
            return 1_000_000_000L;
        return 7_000_000_000L; // Default 7B
    }

    private Uni<InferenceResponse> executeInference(InferenceRequest request) {
        ProviderRequest providerRequest = convertToProviderRequest(request);
        return backend.infer(providerRequest);
    }

    private Multi<StreamingInferenceChunk> executeStreamingInference(InferenceRequest request) {
        ProviderRequest providerRequest = convertToProviderRequest(request);
        return backend.inferStream(providerRequest);
    }

    private ProviderRequest convertToProviderRequest(InferenceRequest request) {
        return ProviderRequest.builder()
                .requestId(request.getRequestId())
                .model(modelPath)
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .metadata("safetensor.backend", providerConfig.backend())
                .metadata("safetensor.basePath", providerConfig.basePath())
                .build();
    }

    private void releaseModel() {
        LOG.debugf("Releasing Safetensor session resources for %s", sessionId);
        // DirectSafetensorBackend handles shared resource management,
        // but session-specific cleanup can happen here if needed.
    }
}
