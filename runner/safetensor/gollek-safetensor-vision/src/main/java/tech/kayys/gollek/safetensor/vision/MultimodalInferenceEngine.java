/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * MultimodalInferenceEngine.java
 * ────────────────────────────────
 * Orchestrates image encoding + LLM generation for vision-language models.
 *
 * Pipeline for LLaVA-1.5
 * ═══════════════════════
 *   User message: "Describe this image" + base64 JPEG
 *
 *   1. Extract images from the message content array
 *   2. VisionEncoder.encode() → patch embeddings [N, lmmDim]
 *   3. Tokenize the text portions of the message
 *   4. Interleave: text prefix + image tokens + text suffix
 *      (replace <image> token in the prompt with image embeddings)
 *   5. Run DirectForwardPass.prefillWithEmbeddings() using combined embeddings
 *   6. Decode normally
 *
 * OpenAI-compatible image_url format
 * ═══════════════════════════════════
 * The REST layer already accepts:
 *   {"role":"user","content":[
 *     {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
 *     {"type":"text","text":"Describe this image"}
 *   ]}
 *
 * This engine handles the multimodal content extraction and routing.
 *
 * Supported models
 * ════════════════
 *   LLaVA-1.5-7B, LLaVA-1.5-13B       (LLaMA-2 + CLIP-ViT-L/14@336)
 *   LLaVA-NeXT-8B, LLaVA-NeXT-34B      (LLaMA-3 + CLIP)
 *   Qwen-VL, Qwen2-VL                   (Qwen + ViT-BigG)
 *   LLaMA-3.2-11B-Vision                (LLaMA-3.1 + vision adapter)
 *   Phi-3-Vision                         (Phi-3 + CLIP)
 *   InternVL2-1B/2B/4B/8B               (InternLM + InternViT)
 */
package tech.kayys.gollek.safetensor.vision;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.safetensor.spi.SafetensorFeature;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.TokenSampler;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.models.ModelArchitectureRegistry;
import tech.kayys.gollek.safetensor.spi.EncodedInput;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;

import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Vision-language model inference engine.
 *
 * <p>
 * Usage:
 * 
 * <pre>{@code
 * @Inject
 * MultimodalInferenceEngine vlmEngine;
 *
 * // Load a VLM model (loads both LLM and vision tower weights)
 * vlmEngine.loadModel(Path.of("/models/llava-1.5-7b"));
 *
 * // Generate with an image
 * Uni<InferenceResponse> resp = vlmEngine.generate(
 *         "data:image/jpeg;base64,...",
 *         "Describe this image in detail.",
 *         modelPath, cfg);
 * }</pre>
 */
@ApplicationScoped
public class MultimodalInferenceEngine implements SafetensorFeature {

    private static final Logger log = Logger.getLogger(MultimodalInferenceEngine.class);

    /** Base64 image URL pattern: data:image/{format};base64,{data} */
    private static final Pattern BASE64_IMAGE_PATTERN = Pattern
            .compile("data:image/(jpeg|png|webp|gif);base64,([A-Za-z0-9+/=]+)");

    @Inject
    DirectInferenceEngine engine;
    @Inject
    VisionEncoder visionEncoder;
    @Inject
    DirectForwardPass forwardPass;
    @Inject
    KVCacheManager kvCacheManager;
    @Inject
    VisionModelRegistry visionRegistry;
    @Inject
    tech.kayys.gollek.safetensor.engine.fusion.EarlyFusionEngine fusionEngine;
    @Inject
    ModelArchitectureRegistry archRegistry;
    @Inject
    TokenSampler sampler;

    @Override
    public String id() {
        return "vision";
    }

    @Override
    public void initialize() {
        log.info("MultimodalInferenceEngine: vision feature registered");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Load
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load a VLM model (LLM backbone + vision tower).
     * Vision tower weights are expected in the same directory as the LLM.
     */
    public void loadModel(Path modelPath) {
        engine.loadModel(modelPath);
        log.infof("MultimodalInferenceEngine: VLM ready [%s]", modelPath.getFileName());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Generate
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Generate text from an image + text prompt.
     *
     * @param imageDataUrl data URL or URL string for the image
     * @param textPrompt   text prompt (may contain {@code <image>} placeholder)
     * @param modelPath    loaded VLM model path
     * @param cfg          generation config
     * @return generated text response
     */
    public Uni<InferenceResponse> generate(String imageDataUrl, String textPrompt,
            Path modelPath, GenerationConfig cfg) {
        return Uni.createFrom().<InferenceResponse>item(() -> {
            Instant t0 = Instant.now();
            try {
                SafetensorEngine.LoadedModel model = requireModel(modelPath);
                Tokenizer tokenizer = model.tokenizer();
                ModelArchitecture arch = archRegistry.resolve(model.config());

                // ── 1. Encode image ───────────────────────────────────────────
                byte[] imageBytes = decodeImageUrl(imageDataUrl);
                String format = detectFormat(imageDataUrl);
                VisionEncoder.VisionConfig visionCfg = visionRegistry.resolve(model.config().modelType(),
                        model.config().hiddenSize());

                VisionEncoder.ImageEmbedding imgEmb = visionEncoder.encode(imageBytes, format,
                        ((Map<String, AccelTensor>)(Map<?, ?>)model.weights()), visionCfg);

                // ── 2. Tokenize text prompt ───────────────────────────────────
                String prompt = visionRegistry.buildPrompt(model.config().modelType(), textPrompt);
                long[] textIds = tokenizer.encode(prompt, EncodeOptions.builder().addBos(true).build());

                log.debugf("VLM: %d image tokens + %d text tokens",
                        imgEmb.numPatches(), textIds.length);

                // ── 3. Fusion ─────────────────────────────────────────────────
                List<EncodedInput> inputs = new ArrayList<>();
                
                // Lookup text embeddings
                AccelTensor embedTable = ((Map<String, AccelTensor>)(Map<?, ?>)model.weights()).get(arch.embedTokensWeight());
                AccelTensor textEmbeds = forwardPass.embeddingLookup(embedTable, textIds);
                float[] textRaw = textEmbeds.toFloatArray();
                float[][] textEmbArray = new float[textIds.length][model.config().hiddenSize()];
                for (int i = 0; i < textIds.length; i++) {
                    System.arraycopy(textRaw, i * model.config().hiddenSize(), textEmbArray[i], 0, model.config().hiddenSize());
                }
                textEmbeds.close();
                
                inputs.add(new EncodedInput(ModalityType.TEXT, textEmbArray, textIds.length, model.config().hiddenSize()));
                
                // Convert Image AccelTensor to float[][] for fusion
                float[] rawEmb = imgEmb.embeddings().toFloatArray();
                float[][] imgEmbArray = new float[imgEmb.numPatches()][imgEmb.llmDim()];
                for (int i = 0; i < imgEmb.numPatches(); i++) {
                    System.arraycopy(rawEmb, i * imgEmb.llmDim(), imgEmbArray[i], 0, imgEmb.llmDim());
                }
                inputs.add(new EncodedInput(ModalityType.IMAGE, imgEmbArray, imgEmb.numPatches(), model.config().hiddenSize()));
                
                tech.kayys.gollek.safetensor.spi.FusionEngine.FusionResult fusion = fusionEngine.fuse(inputs, model.config(), null);

                // ── 4. Prefill ────────────────────────────────────────────────
                KVCacheManager.KVCacheSession kvSession = kvCacheManager.createSession(model.config().maxPositionEmbeddings());
                
                // Convert fused embeddings to AccelTensor
                float[] flatEmbeds = flatten(fusion.embeddings());
                AccelTensor fusedTensor = AccelTensor.fromFloatArray(flatEmbeds, 
                        new long[] { 1, fusion.totalTokens(), model.config().hiddenSize() });
                
                float[] logits = forwardPass.prefill(fusedTensor, textIds, ((Map<String, AccelTensor>)(Map<?, ?>)model.weights()), model.config(), arch, kvSession);
                fusedTensor.close();
                imgEmb.close();

                // ── 5. Decoding Loop ──────────────────────────────────────────
                StringBuilder out = new StringBuilder();
                int nextToken = sampler.sample(logits, cfg, new int[0]);
                int startPos = fusion.totalTokens();
                int generated = 0;
                int maxTokens = cfg.maxNewTokens() > 0 ? cfg.maxNewTokens() : 128;

                while (nextToken != tokenizer.eosTokenId() && generated < maxTokens) {
                    String piece = tokenizer.decode(new long[] { nextToken }, DecodeOptions.defaultOptions());
                    out.append(piece);
                    generated++;

                    logits = forwardPass.decode(nextToken, startPos, ((Map<String, AccelTensor>)(Map<?, ?>)model.weights()), model.config(), arch, kvSession);
                    nextToken = sampler.sample(logits, cfg, new int[0]);
                    startPos++;
                }

                kvSession.close();

                return InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(fusion.totalTokens())
                        .outputTokens(generated)
                        .durationMs(Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(generated >= maxTokens ? InferenceResponse.FinishReason.LENGTH : InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "direct-multimodal")
                        .metadata("image_tokens", imgEmb.numPatches())
                        .build();

            } catch (Exception e) {
                log.error("VLM generation failed", e);
                throw new RuntimeException("VLM generation failed: " + e.getMessage(), e);
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────


    private SafetensorEngine.LoadedModel requireModel(Path path) {
        SafetensorEngine.LoadedModel m = engine.getLoadedModel(path);
        if (m == null)
            throw new IllegalStateException("VLM not loaded: " + path);
        return m;
    }

    private byte[] decodeImageUrl(String url) {
        if (url == null || url.isBlank())
            return new byte[0];
        Matcher m = BASE64_IMAGE_PATTERN.matcher(url);
        if (m.find())
            return Base64.getDecoder().decode(m.group(2));
        // HTTP URL — would need to download; return placeholder for now
        log.warnf("Remote image URL not yet supported: %s", url.substring(0, Math.min(50, url.length())));
        return new byte[0];
    }

    private static String detectFormat(String url) {
        if (url == null)
            return "jpeg";
        if (url.contains("image/png"))
            return "png";
        if (url.contains("image/webp"))
            return "webp";
        return "jpeg";
    }

    private static float[] flatten(float[][] data) {
        if (data == null || data.length == 0) return new float[0];
        int rows = data.length;
        int cols = data[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        }
        return flat;
    }
}
