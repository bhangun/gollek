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

package tech.kayys.gollek.plugin.runner.safetensor.feature.audio;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.plugin.runner.safetensor.feature.SafetensorFeaturePlugin;
import tech.kayys.gollek.safetensor.audio.WhisperEngine;
import tech.kayys.gollek.safetensor.audio.SpeechT5Engine;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.audio.model.AudioResult;
import tech.kayys.gollek.safetensor.audio.processing.AudioProcessor;


import java.util.Map;
import java.util.Set;

/**
 * Audio processing feature plugin for Safetensor.
 * 
 * <p>Integrates with existing Safetensor audio modules:
 * <ul>
 *   <li>{@link WhisperEngine} - Speech-to-text</li>
 *   <li>{@link SpeechT5Engine} - Text-to-speech</li>
 *   <li>{@link AudioProcessor} - Audio processing utilities</li>
 * </ul>
 */
public class AudioFeaturePlugin implements SafetensorFeaturePlugin {

    public static final String ID = "audio-feature";

    private final WhisperEngine whisperEngine;
    private final SpeechT5Engine speechT5Engine;
    private boolean enabled = true;
    private String defaultModel = "whisper-large-v3";
    private String language = "en";

    /**
     * Create audio feature plugin.
     * 
     * @param whisperEngine Whisper engine instance (CDI injected)
     * @param speechT5Engine SpeechT5 engine instance (CDI injected)
     */
    public AudioFeaturePlugin(WhisperEngine whisperEngine, SpeechT5Engine speechT5Engine) {
        this.whisperEngine = whisperEngine;
        this.speechT5Engine = speechT5Engine;
    }

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "Audio Processing";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public String description() {
        return "Audio processing capabilities for speech-to-text, text-to-speech, and audio analysis";
    }

    @Override
    public void initialize(Map<String, Object> config) {
        if (config.containsKey("enabled")) {
            this.enabled = Boolean.parseBoolean(config.get("enabled").toString());
        }
        if (config.containsKey("default_model")) {
            this.defaultModel = config.get("default_model").toString();
        }
        if (config.containsKey("language")) {
            this.language = config.get("language").toString();
        }
    }

    @Override
    public boolean isAvailable() {
        if (!enabled) {
            return false;
        }
        // Check if at least one engine is available
        return (whisperEngine != null) || (speechT5Engine != null);
    }

    @Override
    public int priority() {
        return 100;
    }

    @Override
    public Set<String> supportedModels() {
        return Set.of(
            "whisper-tiny", "whisper-base", "whisper-small", 
            "whisper-medium", "whisper-large", "whisper-large-v3",
            "speecht5-tts", "speecht5-asr"
        );
    }

    @Override
    public Set<String> supportedInputTypes() {
        return Set.of(
            "audio/wav", "audio/mp3", "audio/flac", "audio/ogg",
            "audio/webm", "text/plain"
        );
    }

    @Override
    public Set<String> supportedOutputTypes() {
        return Set.of(
            "text/plain",
            "audio/wav", "audio/mp3"
        );
    }

    @Override
    public Object process(Object input) {
        if (!isAvailable()) {
            throw new IllegalStateException("Audio feature is not available");
        }

        if (input instanceof byte[]) {
            // Audio input - perform speech-to-text
            return transcribeAudio((byte[]) input);
        } else if (input instanceof String) {
            // Text input - perform text-to-speech
            return synthesizeSpeech((String) input);
        } else if (input instanceof AudioInput) {
            return processAudioInput((AudioInput) input);
        } else {
            throw new IllegalArgumentException("Unsupported input type: " + input.getClass());
        }
    }

    @Override
    public Map<String, Object> metadata() {
        return Map.of(
            "type", "audio",
            "whisper_available", whisperEngine != null,
            "speecht5_available", speechT5Engine != null,
            "supported_languages", Set.of("en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"),
            "supported_tasks", Set.of("transcribe", "translate", "synthesize"),
            "max_audio_length_sec", 3600,
            "supported_formats", Set.of("wav", "mp3", "flac", "ogg", "webm")
        );
    }

    @Override
    public void shutdown() {
        enabled = false;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    private Uni<AudioResult> transcribeAudio(byte[] audioData) {
        if (whisperEngine == null) {
            return Uni.createFrom().failure(new IllegalStateException("Whisper engine not available"));
        }
        AudioConfig config = AudioConfig.builder()
                .language(language)
                .task(AudioConfig.Task.TRANSCRIBE)
                .build();
        return whisperEngine.transcribe(audioData, java.nio.file.Path.of("models"), config);
    }

    private Uni<byte[]> synthesizeSpeech(String text) {
        if (speechT5Engine == null) {
            return Uni.createFrom().failure(new IllegalStateException("SpeechT5 engine not available"));
        }

        // Use existing SpeechT5Engine implementation
        AudioConfig config = AudioConfig.builder().build();
        return speechT5Engine.synthesize(text, defaultModel, java.nio.file.Path.of("models"), config);
    }

    private Object processAudioInput(AudioInput audioInput) {
        switch (audioInput.task) {
            case "transcribe":
                return transcribeAudio(audioInput.audioData);
            case "translate":
                return whisperEngine != null ?
                    whisperEngine.transcribe(audioInput.audioData,
                            java.nio.file.Path.of("models"),
                            AudioConfig.builder().language("en").task(AudioConfig.Task.TRANSLATE).build()) :
                    Uni.createFrom().failure(new IllegalStateException("Whisper not available"));
            case "synthesize":
                return synthesizeSpeech(audioInput.text);
            default:
                throw new IllegalArgumentException("Unknown task: " + audioInput.task);
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Helper Classes
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Structured audio input for complex processing tasks.
     */
    public static class AudioInput {
        public final byte[] audioData;
        public final String text;
        public final String task;
        public final String language;
        public final String model;

        public AudioInput(byte[] audioData, String text, String task, String language, String model) {
            this.audioData = audioData;
            this.text = text;
            this.task = task;
            this.language = language;
            this.model = model;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private byte[] audioData;
            private String text;
            private String task = "transcribe";
            private String language;
            private String model;

            public Builder audioData(byte[] audioData) {
                this.audioData = audioData;
                return this;
            }

            public Builder text(String text) {
                this.text = text;
                return this;
            }

            public Builder task(String task) {
                this.task = task;
                return this;
            }

            public Builder language(String language) {
                this.language = language;
                return this;
            }

            public Builder model(String model) {
                this.model = model;
                return this;
            }

            public AudioInput build() {
                return new AudioInput(audioData, text, task, language, model);
            }
        }
    }
}
