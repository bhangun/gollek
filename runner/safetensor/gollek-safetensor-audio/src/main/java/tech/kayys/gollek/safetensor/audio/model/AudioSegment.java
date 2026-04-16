/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioSegment.java
 * ───────────────────────
 * Audio segment with timestamps.
 */
package tech.kayys.gollek.safetensor.audio.model;

import java.util.List;
import java.util.Objects;

/**
 * Represents an audio segment with timing information.
 * Used for transcription segments and speaker diarization.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class AudioSegment {

    /**
     * Segment ID.
     */
    private final int id;

    /**
     * Start time in seconds.
     */
    private final double start;

    /**
     * End time in seconds.
     */
    private final double end;

    /**
     * Transcribed text.
     */
    private final String text;

    /**
     * Speaker ID (for diarization).
     */
    private final String speaker;

    /**
     * Confidence score.
     */
    private final float confidence;

    /**
     * Word-level timestamps.
     */
    private final List<WordTimestamp> words;

    /**
     * Audio features (optional).
     */
    private final float[] features;

    private AudioSegment(Builder builder) {
        this.id = builder.id;
        this.start = builder.start;
        this.end = builder.end;
        this.text = builder.text;
        this.speaker = builder.speaker;
        this.confidence = builder.confidence;
        this.words = builder.words;
        this.features = builder.features;
    }

    /**
     * Create builder for AudioSegment.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create simple segment without word timestamps.
     *
     * @param id     segment ID
     * @param start  start time
     * @param end    end time
     * @param text   transcribed text
     * @param speaker speaker ID
     * @return audio segment
     */
    public static AudioSegment of(int id, double start, double end, String text, String speaker) {
        return builder()
                .id(id)
                .start(start)
                .end(end)
                .text(text)
                .speaker(speaker)
                .build();
    }

    public int getId() {
        return id;
    }

    public double getStart() {
        return start;
    }

    public double getEnd() {
        return end;
    }

    public String getText() {
        return text;
    }

    public String getSpeaker() {
        return speaker;
    }

    public float getConfidence() {
        return confidence;
    }

    public List<WordTimestamp> getWords() {
        return words;
    }

    public float[] getFeatures() {
        return features;
    }

    /**
     * Get segment duration in seconds.
     *
     * @return duration
     */
    public double getDuration() {
        return end - start;
    }

    /**
     * Builder for AudioSegment.
     */
    public static class Builder {
        private int id;
        private double start;
        private double end;
        private String text = "";
        private String speaker = "SPEAKER_00";
        private float confidence = 1.0f;
        private List<WordTimestamp> words = List.of();
        private float[] features;

        public Builder id(int id) {
            this.id = id;
            return this;
        }

        public Builder start(double start) {
            this.start = start;
            return this;
        }

        public Builder end(double end) {
            this.end = end;
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder speaker(String speaker) {
            this.speaker = speaker;
            return this;
        }

        public Builder confidence(float confidence) {
            this.confidence = confidence;
            return this;
        }

        public Builder words(List<WordTimestamp> words) {
            this.words = words;
            return this;
        }

        public Builder features(float[] features) {
            this.features = features;
            return this;
        }

        public AudioSegment build() {
            return new AudioSegment(this);
        }
    }

    /**
     * Word-level timestamp.
     */
    public record WordTimestamp(
            String word,
            double start,
            double end,
            float probability) {
    }
}
