/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.api;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.resteasy.reactive.multipart.FileUpload;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * OpenAI-compatible audio transcription API using Whisper.
 * Uses reflection to access WhisperEngine to avoid circular dependencies.
 */
@jakarta.ws.rs.Path("/v1/audio")
@Tag(name = "Audio", description = "Speech-to-text and text-to-speech via Whisper")
public class AudioResource {

    private static final Logger log = Logger.getLogger(AudioResource.class);

    @Inject
    jakarta.enterprise.inject.Instance<Object> whisperEngineInstance;

    @ConfigProperty(name = "gollek.audio.whisper.default-model", defaultValue = "")
    String defaultWhisperModel;

    private final Map<String, java.nio.file.Path> modelRegistry = new ConcurrentHashMap<>();

    private Object getEngine() {
        try {
            return whisperEngineInstance.get();
        } catch (Exception e) {
            log.error("WhisperEngine not found", e);
            return null;
        }
    }

    @POST
    @jakarta.ws.rs.Path("/transcriptions")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Operation(summary = "Transcribe audio to text (Whisper)")
    public Uni<TranscriptionResponse> transcribe(
            @FormParam("file") FileUpload file,
            @FormParam("model") String model,
            @FormParam("language") String language,
            @FormParam("temperature") Float temperature,
            @FormParam("response_format") String responseFormat,
            @FormParam("prompt") String prompt) {

        java.nio.file.Path modelPath = resolveWhisperModel(model);

        return readFileUpload(file).flatMap(audioBytes -> {
            String format = detectAudioFormat(file);
            Object engine = getEngine();
            if (engine == null) return Uni.createFrom().failure(new RuntimeException("WhisperEngine not available"));

            try {
                // Uni<TranscriptionResult> transcribe(byte[] audioData, String format, Path modelPath, String language, String task)
                Uni<?> uni = (Uni<?>) engine.getClass().getMethod("transcribe", byte[].class, String.class, java.nio.file.Path.class, String.class, String.class)
                        .invoke(engine, audioBytes, format, modelPath, language, "transcribe");
                
                return uni.map(result -> {
                    try {
                        String text = (String) result.getClass().getMethod("text").invoke(result);
                        String lang = (String) result.getClass().getMethod("language").invoke(result);
                        List<?> segments = (List<?>) result.getClass().getMethod("segments").invoke(result);
                        
                        // We need to convert segments safely too, but for simplicity we keep them as raw List for now
                        // or define a simple Segment DTO here.
                        return new TranscriptionResponse(text, lang, segments);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
            } catch (Exception e) {
                return Uni.createFrom().failure(e);
            }
        });
    }

    @POST
    @jakarta.ws.rs.Path("/translations")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Operation(summary = "Translate audio to English text (Whisper translate)")
    public Uni<TranscriptionResponse> translate(
            @FormParam("file") FileUpload file,
            @FormParam("model") String model,
            @FormParam("prompt") String prompt,
            @FormParam("temperature") Float temperature) {

        java.nio.file.Path modelPath = resolveWhisperModel(model);

        return readFileUpload(file).flatMap(bytes -> {
            String format = detectAudioFormat(file);
            Object engine = getEngine();
            if (engine == null) return Uni.createFrom().failure(new RuntimeException("WhisperEngine not available"));

            try {
                Uni<?> uni = (Uni<?>) engine.getClass().getMethod("transcribe", byte[].class, String.class, java.nio.file.Path.class, String.class, String.class)
                        .invoke(engine, bytes, format, modelPath, null, "translate");
                
                return uni.map(r -> {
                    try {
                        String text = (String) r.getClass().getMethod("text").invoke(r);
                        List<?> segments = (List<?>) r.getClass().getMethod("segments").invoke(r);
                        return new TranscriptionResponse(text, "en", segments);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
            } catch (Exception e) {
                return Uni.createFrom().failure(e);
            }
        });
    }

    @POST
    @jakarta.ws.rs.Path("/speech")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces("audio/mpeg")
    @Operation(summary = "Generate speech from text (TTS — coming soon)")
    public byte[] textToSpeech(TtsRequest req) {
        log.warn("TTS: not yet implemented — returning silence placeholder");
        byte[] silence = new byte[24000 * 2 + 44];
        byte[] hdr = buildWavHeader(silence.length - 44, 24000, 1, 16);
        System.arraycopy(hdr, 0, silence, 0, hdr.length);
        return silence;
    }

    @POST
    @jakarta.ws.rs.Path("/models")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    @Operation(summary = "Register a Whisper model alias")
    public Map<String, String> registerModel(AudioModelRegistration req) {
        modelRegistry.put(req.alias(), java.nio.file.Path.of(req.path()));
        return Map.of("alias", req.alias(), "status", "registered");
    }

    private java.nio.file.Path resolveWhisperModel(String modelId) {
        if (modelId == null || modelId.isBlank() || "whisper-1".equals(modelId)) {
            if (!defaultWhisperModel.isBlank())
                return java.nio.file.Path.of(defaultWhisperModel);
            throw new WebApplicationException(
                    jakarta.ws.rs.core.Response.status(400)
                            .entity(Map.of("error", Map.of(
                                    "message", "No Whisper model configured. "
                                            + "Set gollek.audio.whisper.default-model or pass model in request.")))
                            .build());
        }
        java.nio.file.Path p = modelRegistry.get(modelId);
        if (p != null)
            return p;
        java.nio.file.Path direct = java.nio.file.Path.of(modelId);
        if (Files.exists(direct))
            return direct;
        throw new WebApplicationException(
                jakarta.ws.rs.core.Response.status(404)
                        .entity(Map.of("error", Map.of("message", "Whisper model not found: " + modelId))).build());
    }

    private Uni<byte[]> readFileUpload(FileUpload upload) {
        return Uni.createFrom().item(() -> {
            if (upload == null)
                throw new WebApplicationException(
                        jakarta.ws.rs.core.Response.status(400)
                                .entity(Map.of("error", Map.of("message", "file is required"))).build());
            try {
                return Files.readAllBytes(upload.uploadedFile());
            } catch (IOException e) {
                throw new RuntimeException("Failed to read audio file", e);
            }
        });
    }

    private static String detectAudioFormat(FileUpload upload) {
        if (upload == null) return "wav";
        String name = upload.fileName();
        if (name == null) return "wav";
        String lower = name.toLowerCase();
        if (lower.endsWith(".mp3")) return "mp3";
        if (lower.endsWith(".flac")) return "flac";
        if (lower.endsWith(".ogg") || lower.endsWith(".oga")) return "ogg";
        if (lower.endsWith(".m4a") || lower.endsWith(".aac")) return "m4a";
        if (lower.endsWith(".webm")) return "webm";
        return "wav";
    }

    private static byte[] buildWavHeader(int dataLen, int sampleRate, int channels, int bitsPerSample) {
        byte[] h = new byte[44];
        int byteRate = sampleRate * channels * bitsPerSample / 8;
        int blockAlign = channels * bitsPerSample / 8;
        h[0] = 'R'; h[1] = 'I'; h[2] = 'F'; h[3] = 'F';
        intLE(h, 4, dataLen + 36);
        h[8] = 'W'; h[9] = 'A'; h[10] = 'V'; h[11] = 'E';
        h[12] = 'f'; h[13] = 'm'; h[14] = 't'; h[15] = ' ';
        intLE(h, 16, 16);
        shortLE(h, 20, (short) 1);
        shortLE(h, 22, (short) channels);
        intLE(h, 24, sampleRate);
        intLE(h, 28, byteRate);
        shortLE(h, 32, (short) blockAlign);
        shortLE(h, 34, (short) bitsPerSample);
        h[36] = 'd'; h[37] = 'a'; h[38] = 't'; h[39] = 'a';
        intLE(h, 40, dataLen);
        return h;
    }

    private static void intLE(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >> 8);
        b[off + 2] = (byte) (v >> 16);
        b[off + 3] = (byte) (v >> 24);
    }

    private static void shortLE(byte[] b, int off, short v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >> 8);
    }

    public record TranscriptionResponse(
            @JsonProperty("text") String text,
            @JsonProperty("language") String language,
            @JsonProperty("segments") java.util.List<?> segments) {
    }

    public record TtsRequest(
            @JsonProperty("model") String model,
            @JsonProperty("input") String input,
            @JsonProperty("voice") String voice,
            @JsonProperty("speed") Double speed,
            @JsonProperty("response_format") String responseFormat) {
    }

    public record AudioModelRegistration(
            @JsonProperty("alias") String alias,
            @JsonProperty("path") String path) {
    }
}
