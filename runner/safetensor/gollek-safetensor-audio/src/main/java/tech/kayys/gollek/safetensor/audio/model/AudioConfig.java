/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioConfig.java
 * ───────────────────────
 * Audio processing configuration.
 */
package tech.kayys.gollek.safetensor.audio.model;


/**
 * Configuration for audio processing operations.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class AudioConfig {

    /**
     * Audio task type.
     */
    public enum Task {
        TRANSCRIBE,
        TRANSLATE,
        TEXT_TO_SPEECH,
        SPEAKER_VERIFICATION,
        SPEAKER_DIARIZATION
    }

    /**
     * Audio format.
     */
    public enum Format {
        WAV,
        MP3,
        FLAC,
        OGG,
        M4A,
        WEBM
    }

    /**
     * Sample rate in Hz.
     */
    private final int sampleRate;

    /**
     * Number of audio channels (1=mono, 2=stereo).
     */
    private final int channels;

    /**
     * Bits per sample.
     */
    private final int bitsPerSample;

    /**
     * Task type.
     */
    private final Task task;

    /**
     * Language code (ISO-639-1).
     */
    private final String language;

    /**
     * Voice name for TTS.
     */
    private final String voice;

    /**
     * Audio format.
     */
    private final Format format;

    /**
     * Chunk duration in seconds for streaming.
     */
    private final int chunkDurationSec;

    /**
     * Whether to use word-level timestamps.
     */
    private final boolean wordTimestamps;

    /**
     * Beam size for decoding.
     */
    private final int beamSize;

    /**
     * Temperature for sampling.
     */
    private final float temperature;

    /**
     * Whether to detect language automatically.
     */
    private final boolean autoLanguage;

    private AudioConfig(Builder builder) {
        this.sampleRate = builder.sampleRate;
        this.channels = builder.channels;
        this.bitsPerSample = builder.bitsPerSample;
        this.task = builder.task;
        this.language = builder.language;
        this.voice = builder.voice;
        this.format = builder.format;
        this.chunkDurationSec = builder.chunkDurationSec;
        this.wordTimestamps = builder.wordTimestamps;
        this.beamSize = builder.beamSize;
        this.temperature = builder.temperature;
        this.autoLanguage = builder.autoLanguage;
    }

    /**
     * Create builder for AudioConfig.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create default config for transcription.
     *
     * @return transcription configuration
     */
    public static AudioConfig forTranscription() {
        return builder()
                .task(Task.TRANSCRIBE)
                .sampleRate(16000)
                .channels(1)
                .bitsPerSample(16)
                .format(Format.WAV)
                .build();
    }

    /**
     * Create default config for TTS.
     *
     * @param voice voice name
     * @return TTS configuration
     */
    public static AudioConfig forTTS(String voice) {
        return builder()
                .task(Task.TEXT_TO_SPEECH)
                .voice(voice)
                .sampleRate(16000)
                .channels(1)
                .bitsPerSample(16)
                .format(Format.WAV)
                .build();
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public int getChannels() {
        return channels;
    }

    public int getBitsPerSample() {
        return bitsPerSample;
    }

    public Task getTask() {
        return task;
    }

    public String getLanguage() {
        return language;
    }

    public String getVoice() {
        return voice;
    }

    public Format getFormat() {
        return format;
    }

    public int getChunkDurationSec() {
        return chunkDurationSec;
    }

    public boolean isWordTimestamps() {
        return wordTimestamps;
    }

    public int getBeamSize() {
        return beamSize;
    }

    public float getTemperature() {
        return temperature;
    }

    public boolean isAutoLanguage() {
        return autoLanguage;
    }

    /**
     * Builder for AudioConfig.
     */
    public static class Builder {
        private int sampleRate = 16000;
        private int channels = 1;
        private int bitsPerSample = 16;
        private Task task = Task.TRANSCRIBE;
        private String language = "en";
        private String voice = "alloy";
        private Format format = Format.WAV;
        private int chunkDurationSec = 30;
        private boolean wordTimestamps = true;
        private int beamSize = 5;
        private float temperature = 0.0f;
        private boolean autoLanguage = true;

        public Builder sampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public Builder channels(int channels) {
            this.channels = channels;
            return this;
        }

        public Builder bitsPerSample(int bitsPerSample) {
            this.bitsPerSample = bitsPerSample;
            return this;
        }

        public Builder task(Task task) {
            this.task = task;
            return this;
        }

        public Builder language(String language) {
            this.language = language;
            return this;
        }

        public Builder voice(String voice) {
            this.voice = voice;
            return this;
        }

        public Builder format(Format format) {
            this.format = format;
            return this;
        }

        public Builder chunkDurationSec(int chunkDurationSec) {
            this.chunkDurationSec = chunkDurationSec;
            return this;
        }

        public Builder wordTimestamps(boolean wordTimestamps) {
            this.wordTimestamps = wordTimestamps;
            return this;
        }

        public Builder beamSize(int beamSize) {
            this.beamSize = beamSize;
            return this;
        }

        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder autoLanguage(boolean autoLanguage) {
            this.autoLanguage = autoLanguage;
            return this;
        }

        public AudioConfig build() {
            return new AudioConfig(this);
        }
    }
}
