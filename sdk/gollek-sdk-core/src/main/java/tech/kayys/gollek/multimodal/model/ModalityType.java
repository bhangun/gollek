package tech.kayys.gollek.multimodal.model;

/**
 * Enumerates the supported input and output modality types for multimodal requests.
 *
 * <p>Used in {@link MultimodalContent} to identify the kind of data being passed,
 * and in {@link MultimodalRequest.OutputConfig} to specify which modalities the
 * model should produce in its response.
 *
 * @see MultimodalContent
 * @see MultimodalRequest.OutputConfig
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
    TIME_SERIES
}
