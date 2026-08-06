package tech.kayys.alkhawarizm.spi.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;

import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Unified content carrier for a single modality part inside a multimodal request.
 *
 * <p>A {@link MultimodalRequest} is composed of one or more {@code MultimodalContent}
 * instances, each representing a distinct "part" (text snippet, image, audio clip, etc.).
 *
 * <h3>Example – text + image</h3>
 * <pre>{@code
 *   MultimodalContent text  = MultimodalContent.ofText("Describe this image:");
 *   MultimodalContent image = MultimodalContent.ofBase64Image(jpegBytes, "image/jpeg");
 *   MultimodalRequest req   = MultimodalRequest.of(model, List.of(text, image));
 * }</pre>
 */
@JsonInclude(Include.NON_NULL)
public final class MultimodalContent {

    private final ModalityType modality;

    // --- text ---
    private final String text;

    // --- binary (image / audio / video / 3d) ---
    private final byte[]  rawBytes;
    private final String  mimeType;
    private final String  base64Data;   // pre-encoded, mutually exclusive with rawBytes

    // --- uri reference (remote asset) ---
    private final String  uri;

    // --- document ---
    private final String  documentFormat;   // e.g. "pdf", "docx", "html"

    // --- embedding ---
    private final float[] embedding;

    // --- time-series ---
    private final double[] timeSeries;
    private final long     samplingRateHz;

    // --- shared metadata ---
    private final Map<String, Object> metadata;

    private MultimodalContent(Builder b) {
        this.modality       = Objects.requireNonNull(b.modality, "modality");
        this.text           = b.text;
        this.rawBytes       = b.rawBytes;
        this.mimeType       = b.mimeType;
        this.base64Data     = b.base64Data;
        this.uri            = b.uri;
        this.documentFormat = b.documentFormat;
        this.embedding      = b.embedding;
        this.timeSeries     = b.timeSeries;
        this.samplingRateHz = b.samplingRateHz;
        this.metadata       = b.metadata;
    }

    // -------------------------------------------------------------------------
    // Factory shortcuts
    // -------------------------------------------------------------------------

    public static MultimodalContent ofText(String text) {
        return new Builder(ModalityType.TEXT).text(text).build();
    }

    public static MultimodalContent ofBase64Image(byte[] bytes, String mimeType) {
        String b64 = Base64.getEncoder().encodeToString(bytes);
        return new Builder(ModalityType.IMAGE)
                .mimeType(mimeType)
                .base64Data(b64)
                .build();
    }

    public static MultimodalContent ofImageUri(String uri, String mimeType) {
        return new Builder(ModalityType.IMAGE)
                .uri(uri)
                .mimeType(mimeType)
                .build();
    }

    public static MultimodalContent ofAudio(byte[] bytes, String mimeType) {
        String b64 = Base64.getEncoder().encodeToString(bytes);
        return new Builder(ModalityType.AUDIO)
                .mimeType(mimeType)
                .base64Data(b64)
                .build();
    }

    public static MultimodalContent ofAudioUri(String uri) {
        return new Builder(ModalityType.AUDIO).uri(uri).build();
    }

    public static MultimodalContent ofVideo(byte[] bytes, String mimeType) {
        String b64 = Base64.getEncoder().encodeToString(bytes);
        return new Builder(ModalityType.VIDEO)
                .mimeType(mimeType)
                .base64Data(b64)
                .build();
    }

    public static MultimodalContent ofVideoUri(String uri) {
        return new Builder(ModalityType.VIDEO).uri(uri).build();
    }

    public static MultimodalContent ofDocument(byte[] bytes, String format, String mimeType) {
        String b64 = Base64.getEncoder().encodeToString(bytes);
        return new Builder(ModalityType.DOCUMENT)
                .documentFormat(format)
                .mimeType(mimeType)
                .base64Data(b64)
                .build();
    }

    public static MultimodalContent ofDocumentUri(String uri, String format) {
        return new Builder(ModalityType.DOCUMENT)
                .uri(uri)
                .documentFormat(format)
                .build();
    }

    public static MultimodalContent ofEmbedding(float[] embedding) {
        return new Builder(ModalityType.EMBEDDING).embedding(embedding).build();
    }

    public static MultimodalContent ofTimeSeries(double[] data, long samplingRateHz) {
        return new Builder(ModalityType.TIME_SERIES)
                .timeSeries(data)
                .samplingRateHz(samplingRateHz)
                .build();
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public ModalityType getModality()       { return modality; }
    public String       getText()           { return text; }
    public byte[]       getRawBytes()       { return rawBytes; }
    public String       getMimeType()       { return mimeType; }
    public String       getBase64Data()     { return base64Data; }
    public String       getUri()            { return uri; }
    public String       getDocumentFormat() { return documentFormat; }
    public float[]      getEmbedding()      { return embedding; }
    public double[]     getTimeSeries()     { return timeSeries; }
    public long         getSamplingRateHz() { return samplingRateHz; }
    public Map<String, Object> getMetadata() { return metadata; }

    /** Convenience: is this a URI-based (remote) reference? */
    public boolean isRemote() { return uri != null && !uri.isBlank(); }

    // -------------------------------------------------------------------------
    // Builder
    // -------------------------------------------------------------------------

    public static Builder builder(ModalityType modality) {
        return new Builder(modality);
    }

    public static final class Builder {
        private final ModalityType modality;
        private String text;
        private byte[] rawBytes;
        private String mimeType;
        private String base64Data;
        private String uri;
        private String documentFormat;
        private float[] embedding;
        private double[] timeSeries;
        private long samplingRateHz;
        private Map<String, Object> metadata = new HashMap<>();

        private Builder(ModalityType modality) {
            this.modality = modality;
        }

        public Builder text(String v)             { this.text = v; return this; }
        public Builder rawBytes(byte[] v)         { this.rawBytes = v; return this; }
        public Builder mimeType(String v)         { this.mimeType = v; return this; }
        public Builder base64Data(String v)       { this.base64Data = v; return this; }
        public Builder uri(String v)              { this.uri = v; return this; }
        public Builder documentFormat(String v)   { this.documentFormat = v; return this; }
        public Builder embedding(float[] v)       { this.embedding = v; return this; }
        public Builder timeSeries(double[] v)     { this.timeSeries = v; return this; }
        public Builder samplingRateHz(long v)     { this.samplingRateHz = v; return this; }
        public Builder meta(String k, Object v)   { this.metadata.put(k, v); return this; }

        public MultimodalContent build() { return new MultimodalContent(this); }
    }
}
