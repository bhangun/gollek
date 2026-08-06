package tech.kayys.gollek.spi.inference;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;
import java.util.Objects;

/**
 * Multi-modal attachment for inference requests.
 *
 * <p>
 * Supports various content types:
 * <ul>
 * <li>Images (PNG, JPEG, GIF, WEBP)</li>
 * <li>Audio (WAV, MP3, OGG)</li>
 * <li>Documents (PDF, TXT, MD)</li>
 * <li>Videos (MP4, WEBM)</li>
 * </ul>
 *
 * <h3>Usage:</h3>
 * 
 * <pre>{@code
 * // Image from URL
 * Attachment image = Attachment.fromUrl("https://example.com/image.png", "image/png");
 *
 * // Image from base64
 * Attachment image = Attachment.fromBase64(base64Data, "image/png");
 *
 * // Audio file
 * Attachment audio = Attachment.fromUrl("https://example.com/audio.wav", "audio/wav");
 *
 * // Add to request metadata
 * request.getMetadata().put("attachments", List.of(image, audio));
 * }</pre>
 *
 * @author Gollek Team
 * @version 1.0.0
 */
public final class Attachment {

    /**
     * Attachment type enumeration.
     */
    public enum Type {
        IMAGE("image"),
        AUDIO("audio"),
        VIDEO("video"),
        DOCUMENT("document");

        private final String category;

        Type(String category) {
            this.category = category;
        }

        public String category() {
            return category;
        }

        public static Type fromMimeType(String mimeType) {
            if (mimeType == null) {
                return DOCUMENT;
            }

            String type = mimeType.split("/")[0].toLowerCase();
            return switch (type) {
                case "image" -> IMAGE;
                case "audio" -> AUDIO;
                case "video" -> VIDEO;
                default -> DOCUMENT;
            };
        }
    }

    private final String id;
    private final Type type;
    private final String mimeType;
    private final String url;
    private final String base64Data;
    private final Map<String, Object> metadata;

    /**
     * Create an attachment.
     *
     * @param id         unique identifier
     * @param type       attachment type
     * @param mimeType   MIME type
     * @param url        source URL (optional)
     * @param base64Data base64-encoded data (optional)
     * @param metadata   optional metadata
     */
    public Attachment(
            @JsonProperty("id") String id,
            @JsonProperty("type") Type type,
            @JsonProperty("mime_type") String mimeType,
            @JsonProperty("url") String url,
            @JsonProperty("base64_data") String base64Data,
            @JsonProperty("metadata") Map<String, Object> metadata) {

        this.id = Objects.requireNonNull(id, "id");
        this.type = Objects.requireNonNull(type, "type");
        this.mimeType = Objects.requireNonNull(mimeType, "mime_type");
        this.url = url;
        this.base64Data = base64Data;
        this.metadata = metadata != null ? Map.copyOf(metadata) : Map.of();

        // Validate that either URL or base64 data is provided
        if ((url == null || url.isBlank()) && (base64Data == null || base64Data.isBlank())) {
            throw new IllegalArgumentException("Either url or base64_data must be provided");
        }
    }

    /**
     * Create an attachment from URL.
     *
     * @param url      resource URL
     * @param mimeType MIME type
     * @return attachment
     */
    public static Attachment fromUrl(String url, String mimeType) {
        String id = generateId(url);
        Type type = Type.fromMimeType(mimeType);
        return new Attachment(id, type, mimeType, url, null, Map.of());
    }

    /**
     * Create an attachment from base64 data.
     *
     * @param base64Data base64-encoded data
     * @param mimeType   MIME type
     * @return attachment
     */
    public static Attachment fromBase64(String base64Data, String mimeType) {
        String id = generateId(base64Data);
        Type type = Type.fromMimeType(mimeType);
        return new Attachment(id, type, mimeType, null, base64Data, Map.of());
    }

    /**
     * Create an attachment with metadata.
     *
     * @param url      resource URL
     * @param mimeType MIME type
     * @param metadata additional metadata
     * @return attachment
     */
    public static Attachment fromUrl(String url, String mimeType, Map<String, Object> metadata) {
        String id = generateId(url);
        Type type = Type.fromMimeType(mimeType);
        return new Attachment(id, type, mimeType, url, null, metadata);
    }

    // Getters

    public String getId() {
        return id;
    }

    public Type getType() {
        return type;
    }

    public String getMimeType() {
        return mimeType;
    }

    public String getUrl() {
        return url;
    }

    public String getBase64Data() {
        return base64Data;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Check if this attachment has URL source.
     *
     * @return true if URL is present
     */
    public boolean hasUrl() {
        return url != null && !url.isBlank();
    }

    /**
     * Check if this attachment has base64 data.
     *
     * @return true if base64 data is present
     */
    public boolean hasBase64Data() {
        return base64Data != null && !base64Data.isBlank();
    }

    /**
     * Get the file extension from MIME type.
     *
     * @return file extension (e.g., ".png")
     */
    public String getFileExtension() {
        return switch (mimeType.toLowerCase()) {
            case "image/png" -> ".png";
            case "image/jpeg", "image/jpg" -> ".jpg";
            case "image/gif" -> ".gif";
            case "image/webp" -> ".webp";
            case "audio/wav" -> ".wav";
            case "audio/mp3" -> ".mp3";
            case "audio/ogg" -> ".ogg";
            case "video/mp4" -> ".mp4";
            case "video/webm" -> ".webm";
            case "application/pdf" -> ".pdf";
            case "text/plain" -> ".txt";
            case "text/markdown" -> ".md";
            default -> ".bin";
        };
    }

    /**
     * Estimate the size in bytes of this attachment.
     *
     * @return estimated size
     */
    public long estimateSize() {
        if (hasBase64Data()) {
            // Base64 is ~4/3 the original size
            return (long) (base64Data.length() * 0.75);
        }
        // For URL-based, return -1 (unknown)
        return -1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof Attachment))
            return false;
        Attachment that = (Attachment) o;
        return id.equals(that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return String.format("Attachment{id='%s', type=%s, mimeType='%s', hasUrl=%b, hasBase64=%b}",
                id, type, mimeType, hasUrl(), hasBase64Data());
    }

    // Private helpers

    private static String generateId(String source) {
        return "att_" + Integer.toHexString(source.hashCode());
    }
}
