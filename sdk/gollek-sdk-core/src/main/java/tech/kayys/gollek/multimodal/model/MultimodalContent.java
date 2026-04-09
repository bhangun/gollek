package tech.kayys.gollek.multimodal.model;

import java.util.Base64;
import java.util.Map;

/**
 * Represents a single piece of multimodal content — text, image, audio, document,
 * or a pre-computed embedding — that can be included in a {@link MultimodalRequest}.
 *
 * <p>Use the static factory methods for the most common cases, or {@link #builder(ModalityType)}
 * for full control:
 * <pre>{@code
 * MultimodalContent text  = MultimodalContent.ofText("Describe this image");
 * MultimodalContent image = MultimodalContent.ofBase64Image(imageBytes, "image/jpeg");
 * MultimodalContent doc   = MultimodalContent.ofDocument(pdfBytes, "pdf", "application/pdf");
 * }</pre>
 *
 * @see MultimodalRequest
 * @see ModalityType
 */
public final class MultimodalContent {

    private final ModalityType modality;
    private final String text;
    private final String base64Data;
    private final String mimeType;
    private final String uri;
    private final String documentFormat;
    private final float[] embedding;
    private final Map<String, Object> metadata;

    private MultimodalContent(Builder builder) {
        this.modality = builder.modality;
        this.text = builder.text;
        this.base64Data = builder.base64Data;
        this.mimeType = builder.mimeType;
        this.uri = builder.uri;
        this.documentFormat = builder.documentFormat;
        this.embedding = builder.embedding;
        this.metadata = builder.metadata;
    }

    /**
     * Creates a text content item.
     *
     * @param text the plain text value; must not be {@code null}
     * @return a {@link ModalityType#TEXT} content item
     */
    public static MultimodalContent ofText(String text) {
        return builder(ModalityType.TEXT).text(text).build();
    }

    /**
     * Creates an image content item from raw bytes, Base64-encoded internally.
     *
     * @param bytes    raw image bytes
     * @param mimeType MIME type of the image (e.g. {@code "image/jpeg"}, {@code "image/png"})
     * @return a {@link ModalityType#IMAGE} content item
     */
    public static MultimodalContent ofBase64Image(byte[] bytes, String mimeType) {
        String b64 = Base64.getEncoder().encodeToString(bytes);
        return builder(ModalityType.IMAGE).base64Data(b64).mimeType(mimeType).build();
    }

    /**
     * Creates an image content item from a URI (URL or local file path).
     *
     * @param uri      URI pointing to the image resource
     * @param mimeType MIME type of the image
     * @return a {@link ModalityType#IMAGE} content item
     */
    public static MultimodalContent ofImageUri(String uri, String mimeType) {
        return builder(ModalityType.IMAGE).uri(uri).mimeType(mimeType).build();
    }

    /**
     * Creates a document content item from raw bytes, Base64-encoded internally.
     *
     * @param bytes    raw document bytes
     * @param format   document format identifier (e.g. {@code "pdf"}, {@code "docx"})
     * @param mimeType MIME type of the document (e.g. {@code "application/pdf"})
     * @return a {@link ModalityType#DOCUMENT} content item
     */
    public static MultimodalContent ofDocument(byte[] bytes, String format, String mimeType) {
        String b64 = Base64.getEncoder().encodeToString(bytes);
        return builder(ModalityType.DOCUMENT).base64Data(b64).documentFormat(format).mimeType(mimeType).build();
    }

    /**
     * Creates a new builder for the given modality type.
     *
     * @param modality the modality of the content to build
     * @return a new {@link Builder}
     */
    public static Builder builder(ModalityType modality) {
        return new Builder(modality);
    }

    /**
     * Returns the modality type of this content item.
     *
     * @return the {@link ModalityType}
     */
    public ModalityType getModality() {
        return modality;
    }

    /**
     * Returns the plain text value, or {@code null} if this is not a text item.
     *
     * @return text content, or {@code null}
     */
    public String getText() {
        return text;
    }

    /**
     * Returns the Base64-encoded binary data, or {@code null} if not applicable.
     *
     * @return Base64 string, or {@code null}
     */
    public String getBase64Data() {
        return base64Data;
    }

    /**
     * Returns the MIME type of the binary content (e.g. {@code "image/jpeg"}).
     *
     * @return MIME type string, or {@code null}
     */
    public String getMimeType() {
        return mimeType;
    }

    /**
     * Returns the URI pointing to the content resource, or {@code null} if not applicable.
     *
     * @return URI string, or {@code null}
     */
    public String getUri() {
        return uri;
    }

    /**
     * Returns the document format identifier (e.g. {@code "pdf"}), or {@code null}.
     *
     * @return document format, or {@code null}
     */
    public String getDocumentFormat() {
        return documentFormat;
    }

    /**
     * Returns the pre-computed embedding vector, or {@code null} if not applicable.
     *
     * @return float array embedding, or {@code null}
     */
    public float[] getEmbedding() {
        return embedding;
    }

    /**
     * Returns arbitrary key-value metadata attached to this content item.
     *
     * @return metadata map, or {@code null}
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Builder for {@link MultimodalContent}.
     */
    public static final class Builder {
        private final ModalityType modality;
        private String text;
        private String base64Data;
        private String mimeType;
        private String uri;
        private String documentFormat;
        private float[] embedding;
        private Map<String, Object> metadata;

        private Builder(ModalityType modality) {
            this.modality = modality;
        }

        /** @param text plain text value */
        public Builder text(String text) { this.text = text; return this; }

        /** @param base64Data Base64-encoded binary payload */
        public Builder base64Data(String base64Data) { this.base64Data = base64Data; return this; }

        /** @param mimeType MIME type of the binary content */
        public Builder mimeType(String mimeType) { this.mimeType = mimeType; return this; }

        /** @param uri URI pointing to the content resource */
        public Builder uri(String uri) { this.uri = uri; return this; }

        /** @param documentFormat format identifier, e.g. {@code "pdf"} */
        public Builder documentFormat(String documentFormat) { this.documentFormat = documentFormat; return this; }

        /** @param embedding pre-computed dense vector */
        public Builder embedding(float[] embedding) { this.embedding = embedding; return this; }

        /** @param metadata arbitrary key-value pairs */
        public Builder metadata(Map<String, Object> metadata) { this.metadata = metadata; return this; }

        /**
         * Builds the {@link MultimodalContent} instance.
         *
         * @return a new immutable {@link MultimodalContent}
         */
        public MultimodalContent build() {
            return new MultimodalContent(this);
        }
    }
}
