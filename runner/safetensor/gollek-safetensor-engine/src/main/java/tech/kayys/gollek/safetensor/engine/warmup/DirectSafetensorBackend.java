/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.generation.TextInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.models.core.ChatTemplateFormatter;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderResponse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Flow;

/**
 * Direct Safetensor Backend for SafetensorProvider.
 * Delegates requests to the internal native inference engines.
 */
@ApplicationScoped
public class DirectSafetensorBackend {

    private static final Logger log = Logger.getLogger(DirectSafetensorBackend.class);
    private static final String PROVIDER_ID = "safetensor-direct";

    @Inject
    DirectInferenceEngine engine;

    @Inject
    TextInferenceEngine textEngine;

    @Inject
    tech.kayys.gollek.safetensor.audio.SpeechT5Engine speechT5Engine;

    @Inject
    SafetensorProviderConfig config;

    public void initialize(ProviderConfig providerConfig) {
        log.infof("DirectSafetensorBackend: initialised with FFM memory mapped weights");
    }

    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.builder()
                .status(ProviderHealth.Status.HEALTHY)
                .message("Direct Safetensor backend active (zero-copy enabled)")
                .timestamp(Instant.now())
                .build());
    }

    public Uni<InferenceResponse> infer(ProviderRequest request) {
        // For non-streaming, collect everything from the stream until end
        return Multi.createFrom().publisher(streamToPublisher(request))
                .collect().asList()
                .map(chunks -> {
                    long startTimeLocal = System.currentTimeMillis();
                    StringBuilder fullText = new StringBuilder();
                    int totalInputTokens = 0;
                    int totalOutputTokens = 0;
                    long totalDurationMs = 0;
                    String finalRequestId = null;
                    java.util.Map<String, Object> finalMetadata = new java.util.HashMap<>();
                    for (ProviderResponse r : chunks) {
                        if (finalRequestId == null)
                            finalRequestId = r.getRequestId();
                        
                        String delta = r.getContent();
                        String modality = (String) r.getMetadata().get("modality");
                        
                        if ("AUDIO".equals(modality)) {
                            finalMetadata.put("audio", delta);
                        } else if ("IMAGE".equals(modality)) {
                            finalMetadata.put("image", delta);
                        } else if (delta != null) {
                            fullText.append(delta);
                        }

                        // Propagate other metadata
                        finalMetadata.putAll(r.getMetadata());

                        // Pick metrics from chunks (usually the last chunk has total, but we'll sum or
                        // max them)
                        totalInputTokens = Math.max(totalInputTokens, r.getPromptTokens());
                        totalOutputTokens = Math.max(totalOutputTokens, r.getCompletionTokens());
                        totalDurationMs = Math.max(totalDurationMs, r.getDurationMs());
                    }

                    if (totalOutputTokens == 0 && fullText.length() > 0) {
                        totalOutputTokens = fullText.length() / 4; // fallback estimate
                    }

                    return InferenceResponse.builder()
                            .requestId(finalRequestId != null ? finalRequestId : "chatcmpl-" + request.getRequestId())
                            .model(request.getModel())
                            .content(fullText.toString())
                            .tokensUsed(totalInputTokens + totalOutputTokens)
                            .inputTokens(totalInputTokens)
                            .outputTokens(totalOutputTokens)
                            .durationMs(totalDurationMs > 0 ? totalDurationMs : (System.currentTimeMillis() - startTimeLocal))
                            .metadata(finalMetadata)
                            .build();
                });
    }

    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        java.util.concurrent.atomic.AtomicInteger idx = new java.util.concurrent.atomic.AtomicInteger(0);
        return Multi.createFrom().publisher(streamToPublisher(request))
                .map(response -> {
                    boolean isFinal = "stop".equals(response.getFinishReason());
                    int index = idx.getAndIncrement();
                    if ("AUDIO".equals(response.getMetadata().get("modality"))) {
                        return new StreamingInferenceChunk(response.getRequestId(), index, 
                                tech.kayys.gollek.spi.model.ModalityType.AUDIO, 
                                response.getContent(), null, isFinal, response.getFinishReason(), null, Instant.now(), response.getMetadata());
                    }
                    if (isFinal) {
                        return StreamingInferenceChunk.finalChunk(response.getRequestId(), index,
                                response.getContent());
                    } else {
                        return StreamingInferenceChunk.of(response.getRequestId(), index, response.getContent());
                    }
                });
    }

    private Flow.Publisher<ProviderResponse> streamToPublisher(ProviderRequest request) {
        // Resolve model path: prefer explicit model_path parameter/metadata over model ID
        // (model ID may be a HuggingFace hub name like "openai/whisper-large-v3-turbo")
        Path modelPath = resolveModelPathFromRequest(request);
        try {
            // Load the model zero-copy into native memory Arena

            String loadedKey = engine.loadModel(modelPath);

            // Resolve model type for chat template selection
            var loadedModel = engine.getLoadedModel(modelPath);
            String modelType = (loadedModel != null && loadedModel.config() != null)
                    ? loadedModel.config().modelType()
                    : "";
            String arch = (loadedModel != null && loadedModel.config() != null)
                    ? loadedModel.config().primaryArchitecture()
                    : "";
            
            boolean isAudioModel = ("speecht5".equals(modelType) || "whisper".equals(modelType) 
                    || modelPath.toString().contains("speecht5")
                    || (arch != null && (arch.toLowerCase().contains("speecht5") || arch.toLowerCase().contains("whisper"))));
            
            System.err.println("[DEBUG-ROUTING] modelPath: " + modelPath);
            System.err.println("[DEBUG-ROUTING] loadedModel null? " + (loadedModel == null));
            System.err.println("[DEBUG-ROUTING] config null? " + (loadedModel != null && loadedModel.config() == null));
            System.err.println("[DEBUG-ROUTING] modelType: '" + modelType + "'");
            System.err.println("[DEBUG-ROUTING] arch: '" + arch + "'");
            System.err.println("[DEBUG-ROUTING] isAudioModel: " + isAudioModel);


            // Build messages list — inject a default system message if none provided
            List<tech.kayys.gollek.spi.Message> rawMessages = new ArrayList<>(request.getMessages());
            boolean hasSystem = rawMessages.stream()
                    .anyMatch(m -> m.getRole() == tech.kayys.gollek.spi.Message.Role.SYSTEM);
            if (!hasSystem) {
                String defaultSystem = modelType.toLowerCase().contains("qwen")
                    ? "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                    : "You are a helpful assistant.";
                rawMessages.add(0, tech.kayys.gollek.spi.Message.system(defaultSystem));
            }

            // Apply the chat template — use Object bridge to bypass cross-module generic
            // type identity check
            @SuppressWarnings("unchecked")
            String prompt = ChatTemplateFormatter.format((java.util.List) (Object) rawMessages, modelType);
            System.err.println("FORMATTED PROMPT: " + prompt.replace("\n", "\\n"));

            int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : 256;

            String kvQuantStr = request.getParameter("kv_cache_quant", String.class).orElse("none");
            GenerationConfig.KvCacheQuantization kvQuant = GenerationConfig.KvCacheQuantization.NONE;
            try {
                kvQuant = GenerationConfig.KvCacheQuantization.valueOf(kvQuantStr.toUpperCase());
            } catch (Exception ignored) {}

            GenerationConfig genCfg = GenerationConfig.builder()
                    .maxNewTokens(maxTokens)
                    .temperature((float) request.getTemperature())
                    .topP((float) request.getTopP())
                    .repetitionPenalty((float) request.getRepeatPenalty())
                    .kvCacheQuant(kvQuant)
                    .build();

            if (isAudioModel) {
                String ttsPrompt = rawMessages.stream()
                        .filter(m -> m.getRole() == tech.kayys.gollek.spi.Message.Role.USER)
                        .map(tech.kayys.gollek.spi.Message::getContent)
                        .reduce((first, second) -> second)
                        .orElse(prompt);
                
                String outputFormat = request.getParameter("output_format", String.class).orElse("wav");
                tech.kayys.gollek.safetensor.audio.model.AudioConfig.Format format = 
                    tech.kayys.gollek.safetensor.audio.model.AudioConfig.Format.WAV;
                try {
                    format = tech.kayys.gollek.safetensor.audio.model.AudioConfig.Format.valueOf(outputFormat.toUpperCase());
                } catch (Exception e) {
                    // Fallback to WAV or FLAC based on logic
                }

                tech.kayys.gollek.safetensor.audio.model.AudioConfig audioCfg = 
                        tech.kayys.gollek.safetensor.audio.model.AudioConfig.builder()
                        .temperature((float)request.getTemperature())
                        .format(format)
                        .build();

                long synthStart = System.currentTimeMillis();
                int promptTokens = prompt.length() / 4; // fallback estimate if we don't want to re-tokenize
                return Multi.createFrom().publisher(
                        speechT5Engine.synthesize(ttsPrompt, "alloy", modelPath, audioCfg)
                            .map(bytes -> {
                                long durationMs = System.currentTimeMillis() - synthStart;
                                String b64 = java.util.Base64.getEncoder().encodeToString(bytes);
                                return ProviderResponse.builder()
                                        .requestId(request.getRequestId())
                                        .model(request.getModel())
                                        .content(b64)
                                        .finishReason("stop")
                                        .promptTokens(promptTokens)
                                        .completionTokens(bytes.length / 1024) // rough "token" equivalent for audio bytes
                                        .metadata("modality", "AUDIO")
                                        .durationMs(durationMs)
                                        .build();
                            })
                            .toMulti()
                );
            }

            // Generate response lazily
            return textEngine.generateStream(prompt, modelPath, genCfg)
                    .map(infResp -> ProviderResponse.builder()
                            .requestId(infResp.getRequestId())
                            .content(infResp.getContent())
                            .model(infResp.getModel())
                            .finishReason(
                                    infResp.getFinishReason() != null ? infResp.getFinishReason().name().toLowerCase()
                                            : "stop")
                            .promptTokens(infResp.getInputTokens())
                            .completionTokens(infResp.getOutputTokens())
                            .totalTokens(infResp.getTokensUsed())
                            .durationMs(infResp.getDurationMs())
                            .build());
        } catch (Exception e) {
            String msg = (e.getMessage() != null) ? e.getMessage() : "(no message)";
            System.err.println("FATAL in streamToPublisher [" + e.getClass().getSimpleName() + "]: " + msg);
            e.printStackTrace(System.err);
            log.errorf(e, "Direct inference failed for model %s (resolved path: %s)",
                    request.getModel(), modelPath);
            throw new ProviderException("Direct inference failed for model "
                    + request.getModel() + " [" + e.getClass().getSimpleName() + "]: " + msg, e);
        }
    }

    /**
     * Resolve the physical model path from a {@link ProviderRequest}.
     *
     * <p>Lookup order:
     * <ol>
     *   <li>{@code parameters["model_path"]} — set by ChatSessionManager for manifest-based models</li>
     *   <li>{@code metadata["model_path"]}   — injected by ModelRouterService.selectProvider()</li>
     *   <li>{@code request.getModel()}        — raw model ID / absolute path fallback</li>
     * </ol>
     */
    private Path resolveModelPathFromRequest(ProviderRequest request) {
        // 1. Check parameters first (set by ChatSessionManager)
        java.util.Optional<String> fromParam = request.getParameter("model_path", String.class);
        if (fromParam.isPresent() && !fromParam.get().isBlank()) {
            log.debugf("DirectSafetensorBackend: using model_path from parameters: %s", fromParam.get());
            return resolveModelPath(fromParam.get());
        }

        // 2. Check metadata (injected by ModelRouterService)
        Object fromMeta = request.getMetadata().get("model_path");
        if (fromMeta instanceof String s && !s.isBlank()) {
            log.debugf("DirectSafetensorBackend: using model_path from metadata: %s", s);
            return resolveModelPath(s);
        }

        // 3. Fallback to model ID (works for absolute paths or pre-configured aliases)
        log.debugf("DirectSafetensorBackend: no model_path override, resolving from model ID: %s", request.getModel());
        return resolveModelPath(request.getModel());
    }

    /**
     * Resolve a model path string (absolute path, or relative under basePath)
     * to a real filesystem path.
     */
    private Path resolveModelPath(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            throw new ProviderException("Model ID is null or blank");
        }

        Path asPath = Path.of(modelId);

        // If the model ID is already an absolute path, use it directly
        if (asPath.isAbsolute() && Files.exists(asPath)) {
            return asPath;
        }

        // Resolve relative to the configured base path
        Path resolved = Path.of(config.basePath(), modelId);
        if (Files.exists(resolved)) {
            log.debugf("Resolved model '%s' to path: %s", modelId, resolved);
            return resolved;
        }

        // Fall back to the as-given path (will likely fail but gives a clear error)
        log.warnf("Model path not found at %s, using as-is: %s", resolved, modelId);
        return asPath;
    }
}
