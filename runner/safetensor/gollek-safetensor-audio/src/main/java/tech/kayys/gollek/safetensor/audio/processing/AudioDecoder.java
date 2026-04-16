/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioDecoder.java
 * ───────────────────────
 * Audio decoder service interface.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import java.io.IOException;

/**
 * Audio decoder interface for multi-format support.
 * <p>
 * Implementations decode various audio formats to 16kHz mono PCM float array.
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public interface AudioDecoder {

    /**
     * Decode audio bytes to PCM float array.
     *
     * @param audioBytes input audio bytes
     * @return PCM float array normalized to [-1, 1]
     * @throws IOException if decoding fails
     */
    float[] decode(byte[] audioBytes) throws IOException;

    /**
     * Get the supported audio format.
     *
     * @return format name (e.g., "mp3", "flac", "ogg")
     */
    String getFormat();

    /**
     * Check if this decoder can handle the given format.
     *
     * @param format format name or file extension
     * @return true if supported
     */
    boolean supports(String format);
}
