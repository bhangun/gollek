/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioResult.java
 * ───────────────────────
 * Audio processing result.
 */
package tech.kayys.gollek.safetensor.audio.model;

import java.util.List;
import java.util.Map;

/**
 * Result of audio processing operations.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class AudioResult {

    /**
     * Result type.
     */
    public enum ResultType {
        TRANSCRIPTION,
        TRANSLATION,
        SPEECH_SYNTHESIS,
        SPEAKER_EMBEDDING,
        DIARIZATION
    }

    /**
     * Result type.
     */
    private final ResultType type;

    /**
     * Text output (for transcription/translation).
     */
    private final String text;

    /**
     * Audio data (for TTS).
     */
    private final byte[] audioData;

    /**
     * Audio segments.
     */
    private final List<AudioSegment> segments;

    /**
     * Detected language.
     */
    private final String language;

    /**
     * Model used.
     */
    private final String model;

    /**
     * Processing duration in milliseconds.
     */
    private final long durationMs;

    /**
     * Audio duration in seconds.
     */
    private final double audioDurationSec;

    /**
     * Confidence score.
     */
    private final float confidence;

    /**
     * Speaker embeddings.
     */
    private final float[] speakerEmbedding;

    /**
     * Speaker labels for diarization.
     */
    private final Map<Integer, String> speakerLabels;

    /**
     * Additional metadata.
     */
    private final Map<String, Object> metadata;

    /**
     * Whether processing was successful.
     */
    private final boolean success;

    /**
     * Error message if failed.
     */
    private final String errorMessage;

    private AudioResult(Builder builder) {
        this.type = builder.type;
        this.text = builder.text;
        this.audioData = builder.audioData;
        this.segments = builder.segments;
        this.language = builder.language;
        this.model = builder.model;
        this.durationMs = builder.durationMs;
        this.audioDurationSec = builder.audioDurationSec;
        this.confidence = builder.confidence;
        this.speakerEmbedding = builder.speakerEmbedding;
        this.speakerLabels = builder.speakerLabels;
        this.metadata = builder.metadata;
        this.success = builder.success;
        this.errorMessage = builder.errorMessage;
    }

    /**
     * Create builder for AudioResult.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create successful transcription result.
     *
     * @param text     transcribed text
     * @param language detected language
     * @param segments audio segments
     * @param model    model name
     * @return transcription result
     */
    public static AudioResult transcription(String text, String language, List<AudioSegment> segments, String model) {
        return builder()
                .type(ResultType.TRANSCRIPTION)
                .text(text)
                .language(language)
                .segments(segments)
                .model(model)
                .success(true)
                .build();
    }

    /**
     * Create successful TTS result.
     *
     * @param audioData WAV audio bytes
     * @param model     model name
     * @param duration  audio duration in seconds
     * @return speech synthesis result
     */
    public static AudioResult speechSynthesis(byte[] audioData, String model, double duration) {
        return builder()
                .type(ResultType.SPEECH_SYNTHESIS)
                .audioData(audioData)
                .model(model)
                .audioDurationSec(duration)
                .success(true)
                .build();
    }

    /**
     * Create failed result.
     *
     * @param errorMessage error message
     * @return failed result
     */
    public static AudioResult failure(String errorMessage) {
        return builder()
                .success(false)
                .errorMessage(errorMessage)
                .build();
    }

    public ResultType getType() {
        return type;
    }

    public String getText() {
        return text;
    }

    public byte[] getAudioData() {
        return audioData;
    }

    public List<AudioSegment> getSegments() {
        return segments;
    }

    public String getLanguage() {
        return language;
    }

    public String getModel() {
        return model;
    }

    public long getDurationMs() {
        return durationMs;
    }

    public double getAudioDurationSec() {
        return audioDurationSec;
    }

    public float getConfidence() {
        return confidence;
    }

    public float[] getSpeakerEmbedding() {
        return speakerEmbedding;
    }

    public Map<Integer, String> getSpeakerLabels() {
        return speakerLabels;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public boolean isSuccess() {
        return success;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Builder for AudioResult.
     */
    public static class Builder {
        private ResultType type = ResultType.TRANSCRIPTION;
        private String text;
        private byte[] audioData;
        private List<AudioSegment> segments = List.of();
        private String language = "en";
        private String model = "whisper";
        private long durationMs;
        private double audioDurationSec;
        private float confidence = 1.0f;
        private float[] speakerEmbedding;
        private Map<Integer, String> speakerLabels = Map.of();
        private Map<String, Object> metadata = Map.of();
        private boolean success = true;
        private String errorMessage;

        public Builder type(ResultType type) {
            this.type = type;
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder audioData(byte[] audioData) {
            this.audioData = audioData;
            return this;
        }

        public Builder segments(List<AudioSegment> segments) {
            this.segments = segments;
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

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
            return this;
        }

        public Builder audioDurationSec(double audioDurationSec) {
            this.audioDurationSec = audioDurationSec;
            return this;
        }

        public Builder confidence(float confidence) {
            this.confidence = confidence;
            return this;
        }

        public Builder speakerEmbedding(float[] speakerEmbedding) {
            this.speakerEmbedding = speakerEmbedding;
            return this;
        }

        public Builder speakerLabels(Map<Integer, String> speakerLabels) {
            this.speakerLabels = speakerLabels;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder success(boolean success) {
            this.success = success;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public AudioResult build() {
            return new AudioResult(this);
        }
    }
}
