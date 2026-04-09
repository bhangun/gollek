package tech.kayys.gollek.sdk.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Unified content part for building multimodal requests at the SDK level.
 * <p>
 * Wraps the SPI-level {@link MultimodalContent} with convenience factory
 * methods that handle file I/O, MIME type detection, and metadata tagging.
 *
 * <h3>Examples</h3>
 * <pre>{@code
 * // From a file (auto-detects MIME type)
 * ContentPart image = ContentPart.fromFile(Path.of("photo.jpg"));
 *
 * // Text
 * ContentPart text = ContentPart.text("Describe this image");
 *
 * // From URL
 * ContentPart remote = ContentPart.fromUrl("https://example.com/img.png", ModalityType.IMAGE);
 *
 * // Audio bytes
 * ContentPart audio = ContentPart.audio(wavBytes, "audio/wav");
 * }</pre>
 */
public final class ContentPart {

    private final MultimodalContent content;

    private ContentPart(MultimodalContent content) {
        this.content = Objects.requireNonNull(content, "content");
    }

    /**
     * Returns the underlying SPI content object.
     */
    public MultimodalContent toSpiContent() {
        return content;
    }

    /**
     * Returns the modality of this content part.
     */
    public ModalityType modality() {
        return content.getModality();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Factory Methods — Text
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates a text content part.
     *
     * @param text the text content
     * @return a TEXT-modality content part
     */
    public static ContentPart text(String text) {
        return new ContentPart(MultimodalContent.ofText(text));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Factory Methods — Image
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates an image content part from raw bytes.
     *
     * @param bytes    image bytes (JPEG, PNG, WebP, GIF)
     * @param mimeType MIME type (e.g., "image/jpeg")
     * @return an IMAGE-modality content part
     */
    public static ContentPart image(byte[] bytes, String mimeType) {
        return new ContentPart(MultimodalContent.ofBase64Image(bytes, mimeType));
    }

    /**
     * Creates an image content part from a file path.
     * Auto-detects MIME type from the file extension.
     *
     * @param path path to the image file
     * @return an IMAGE-modality content part
     * @throws IOException if the file cannot be read
     */
    public static ContentPart image(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        String mime = detectMimeType(path, "image/");
        return image(bytes, mime);
    }

    /**
     * Creates an image content part from a URL reference.
     *
     * @param url      URL to the image
     * @param mimeType MIME type
     * @return an IMAGE-modality content part
     */
    public static ContentPart imageUrl(String url, String mimeType) {
        return new ContentPart(MultimodalContent.ofImageUri(url, mimeType));
    }

    /**
     * Creates an image content part from a URL reference with auto-detected MIME type.
     *
     * @param url URL to the image
     * @return an IMAGE-modality content part
     */
    public static ContentPart imageUrl(String url) {
        String mime = guessMimeFromExtension(url, "image/jpeg");
        return imageUrl(url, mime);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Factory Methods — Audio
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates an audio content part from raw bytes.
     *
     * @param bytes    audio bytes (WAV, MP3, FLAC, OGG)
     * @param mimeType MIME type (e.g., "audio/wav")
     * @return an AUDIO-modality content part
     */
    public static ContentPart audio(byte[] bytes, String mimeType) {
        return new ContentPart(MultimodalContent.ofAudio(bytes, mimeType));
    }

    /**
     * Creates an audio content part from a file path.
     *
     * @param path path to the audio file
     * @return an AUDIO-modality content part
     * @throws IOException if the file cannot be read
     */
    public static ContentPart audio(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        String mime = detectMimeType(path, "audio/");
        return audio(bytes, mime);
    }

    /**
     * Creates an audio content part from a URL reference.
     *
     * @param url URL to the audio resource
     * @return an AUDIO-modality content part
     */
    public static ContentPart audioUrl(String url) {
        return new ContentPart(MultimodalContent.ofAudioUri(url));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Factory Methods — Video
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates a video content part from raw bytes.
     *
     * @param bytes    video bytes (MP4, AVI, MOV)
     * @param mimeType MIME type (e.g., "video/mp4")
     * @return a VIDEO-modality content part
     */
    public static ContentPart video(byte[] bytes, String mimeType) {
        return new ContentPart(MultimodalContent.ofVideo(bytes, mimeType));
    }

    /**
     * Creates a video content part from a file path.
     *
     * @param path path to the video file
     * @return a VIDEO-modality content part
     * @throws IOException if the file cannot be read
     */
    public static ContentPart video(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        String mime = detectMimeType(path, "video/");
        return video(bytes, mime);
    }

    /**
     * Creates a video content part from a URL reference.
     *
     * @param url URL to the video resource
     * @return a VIDEO-modality content part
     */
    public static ContentPart videoUrl(String url) {
        return new ContentPart(MultimodalContent.ofVideoUri(url));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Factory Methods — Document
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates a document content part from a file path.
     *
     * @param path   path to the document (PDF, DOCX, HTML)
     * @param format document format (e.g., "pdf", "docx")
     * @return a DOCUMENT-modality content part
     * @throws IOException if the file cannot be read
     */
    public static ContentPart document(Path path, String format) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        String mime = detectMimeType(path, "application/");
        return new ContentPart(MultimodalContent.ofDocument(bytes, format, mime));
    }

    /**
     * Creates a document content part from a URL reference.
     *
     * @param url    URL to the document
     * @param format document format (e.g., "pdf")
     * @return a DOCUMENT-modality content part
     */
    public static ContentPart documentUrl(String url, String format) {
        return new ContentPart(MultimodalContent.ofDocumentUri(url, format));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Factory Methods — Generic file
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates a content part from any file, auto-detecting the modality
     * from the file extension.
     *
     * @param path path to the file
     * @return a content part with auto-detected modality
     * @throws IOException if the file cannot be read
     */
    public static ContentPart fromFile(Path path) throws IOException {
        String ext = getExtension(path.getFileName().toString()).toLowerCase();
        return switch (ext) {
            case "jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "svg" -> image(path);
            case "wav", "mp3", "flac", "ogg", "aac", "m4a", "wma" -> audio(path);
            case "mp4", "avi", "mov", "mkv", "webm", "flv" -> video(path);
            case "pdf" -> document(path, "pdf");
            case "docx" -> document(path, "docx");
            case "html", "htm" -> document(path, "html");
            case "txt", "md", "csv", "json", "xml", "yaml", "yml" ->
                    text(Files.readString(path));
            default -> throw new IllegalArgumentException(
                    "Cannot auto-detect modality for extension: ." + ext +
                    ". Use an explicit factory method instead.");
        };
    }

    /**
     * Creates a content part from a URL, auto-detecting modality.
     *
     * @param url      URL to the resource
     * @param modality the content modality
     * @return a content part for the given modality
     */
    public static ContentPart fromUrl(String url, ModalityType modality) {
        return switch (modality) {
            case IMAGE -> imageUrl(url);
            case AUDIO -> audioUrl(url);
            case VIDEO -> videoUrl(url);
            case DOCUMENT -> documentUrl(url, "unknown");
            default -> throw new IllegalArgumentException(
                    "URL-based loading not supported for modality: " + modality);
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MIME type detection
    // ═══════════════════════════════════════════════════════════════════════

    private static final Map<String, String> MIME_MAP = Map.ofEntries(
            // Images
            Map.entry("jpg", "image/jpeg"), Map.entry("jpeg", "image/jpeg"),
            Map.entry("png", "image/png"), Map.entry("gif", "image/gif"),
            Map.entry("webp", "image/webp"), Map.entry("bmp", "image/bmp"),
            Map.entry("svg", "image/svg+xml"), Map.entry("tiff", "image/tiff"),
            // Audio
            Map.entry("wav", "audio/wav"), Map.entry("mp3", "audio/mpeg"),
            Map.entry("flac", "audio/flac"), Map.entry("ogg", "audio/ogg"),
            Map.entry("aac", "audio/aac"), Map.entry("m4a", "audio/mp4"),
            // Video
            Map.entry("mp4", "video/mp4"), Map.entry("avi", "video/x-msvideo"),
            Map.entry("mov", "video/quicktime"), Map.entry("mkv", "video/x-matroska"),
            Map.entry("webm", "video/webm"),
            // Documents
            Map.entry("pdf", "application/pdf"),
            Map.entry("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            Map.entry("html", "text/html"), Map.entry("htm", "text/html")
    );

    private static String detectMimeType(Path path, String fallbackPrefix) {
        String ext = getExtension(path.getFileName().toString()).toLowerCase();
        return MIME_MAP.getOrDefault(ext, fallbackPrefix + "octet-stream");
    }

    private static String guessMimeFromExtension(String url, String fallback) {
        int dot = url.lastIndexOf('.');
        if (dot < 0) return fallback;
        String ext = url.substring(dot + 1).split("[?#]")[0].toLowerCase();
        return MIME_MAP.getOrDefault(ext, fallback);
    }

    private static String getExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot >= 0 ? filename.substring(dot + 1) : "";
    }

    @Override
    public String toString() {
        return "ContentPart{modality=" + modality() + "}";
    }
}
