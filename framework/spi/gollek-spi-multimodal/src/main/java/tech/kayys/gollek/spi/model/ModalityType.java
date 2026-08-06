package tech.kayys.alkhawarizm.spi.model;

/**
 * Enumerates the supported input and output modality types for multimodal requests.
 *
 * <p>This is the canonical enum used across the entire Gollek platform. All SDK, CLI,
 * runner, and plugin components should reference this type from the SPI module.
 *
 * <p>Used in {@link tech.kayys.alkhawarizm.spi.model.MultimodalContent} to identify the kind
 * of data being passed, and in inference requests to specify which modalities the
 * model should produce in its response.
 *
 * @see tech.kayys.alkhawarizm.spi.model.MultimodalContent
 */
public enum ModalityType {
    /** Plain text input or output. */
    TEXT,
    /** Still image (JPEG, PNG, WebP, etc.). */
    IMAGE,
    /** Audio clip (WAV, MP3, etc.). */
    AUDIO,
    /** Video clip. */
    VIDEO,
    /** Structured document (PDF, DOCX, etc.). */
    DOCUMENT,
    /** Pre-computed dense vector embedding. */
    EMBEDDING,
    /** Numeric time-series data. */
    TIME_SERIES;

    /**
     * Returns true if this modality represents binary data (non-text).
     * @return true for IMAGE, AUDIO, VIDEO, DOCUMENT, EMBEDDING, TIME_SERIES
     */
    public boolean isBinary() {
        return this != TEXT;
    }
}
