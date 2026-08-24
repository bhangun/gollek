package tech.kayys.gollek.spi.audio;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;

/**
 * Top-level SPI pipeline contract for Audio Intelligence (TTS, STT, and Realtime Whisper Streaming).
 */
public interface AudioPipeline {

    String id();

    /** Synthesize text to speech */
    Uni<TtsResult> synthesize(TtsRequest request);

    /** Transcribe audio to text (batch) */
    Uni<SttResult> transcribe(SttRequest request);

    /**
     * Realtime streaming transcription for audio chunks (e.g. streaming mic input to live tokens).
     */
    Multi<SttResult.Segment> streamTranscribe(Multi<byte[]> audioStream, SttRequest initialRequest);
}
