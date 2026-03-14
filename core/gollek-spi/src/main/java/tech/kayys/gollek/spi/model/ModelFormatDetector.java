package tech.kayys.gollek.spi.model;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;

/**
 * Stateless utility for detecting {@link ModelFormat} from a file path.
 *
 * <p>Detection strategy (in priority order):
 * <ol>
 *   <li>Magic-byte sniff — reads at most 9 bytes from the file header.</li>
 *   <li>File-extension match — fast path when the file cannot be opened.</li>
 * </ol>
 *
 * <p>Both GGUF and SafeTensors have well-defined magic values:
 * <ul>
 *   <li>GGUF — {@code 0x47475546} (ASCII "GGUF") at offset 0, little-endian uint32.</li>
 *   <li>SafeTensors — the first 8 bytes encode a little-endian uint64 that is the
 *       length of the JSON header; the JSON always starts with {@code {"}, so byte
 *       9 (when the header length &gt; 0) produces a reliable secondary signal.</li>
 * </ul>
 */
public final class ModelFormatDetector {

    // ── GGUF ────────────────────────────────────────────────────────────────
    /** ASCII "GGUF" as a little-endian uint32. */
    private static final int GGUF_MAGIC = 0x46554747; // 'G','G','U','F'

    // ── SafeTensors ─────────────────────────────────────────────────────────
    /** Minimum plausible header-length value for the smallest possible JSON header. */
    private static final long SAFETENSORS_MIN_HEADER_LEN = 8L;
    /** Maximum reasonable SafeTensors header (256 MiB) to guard against garbage. */
    private static final long SAFETENSORS_MAX_HEADER_LEN = 256L * 1024 * 1024;

    private ModelFormatDetector() {}

    // ── Public API ───────────────────────────────────────────────────────────

    /**
     * Detect the format of the model file at {@code path}.
     *
     * @param path absolute or relative path to the model file
     * @return detected format, or {@link Optional#empty()} when unrecognised
     */
    public static Optional<ModelFormat> detect(Path path) {
        if (path == null || !Files.isRegularFile(path)) {
            return Optional.empty();
        }

        // 1. Try magic-byte detection (most authoritative)
        Optional<ModelFormat> byMagic = detectByMagic(path);
        if (byMagic.isPresent()) {
            return byMagic;
        }

        // 2. Fall back to extension
        return detectByExtension(path.getFileName().toString());
    }

    /**
     * Lightweight extension-only check — no I/O.
     *
     * @param fileName file name or full path string
     * @return detected format, or {@link Optional#empty()} when unrecognised
     */
    public static Optional<ModelFormat> detectByExtension(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return Optional.empty();
        }
        String lower = fileName.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".gguf")) {
            return Optional.of(ModelFormat.GGUF);
        }
        if (lower.endsWith(".safetensors") || lower.endsWith(".safetensor")) {
            return Optional.of(ModelFormat.SAFETENSORS);
        }
        if (lower.endsWith(".onnx")) {
            return Optional.of(ModelFormat.ONNX);
        }
        if (lower.endsWith(".tflite")) {
            return Optional.of(ModelFormat.LITERT);
        }
        return Optional.empty();
    }

    /**
     * Check whether {@code path} is a GGUF file.
     *
     * @param path path to inspect
     * @return true if the file is a valid GGUF file
     */
    public static boolean isGguf(Path path) {
        return detect(path).map(f -> f == ModelFormat.GGUF).orElse(false);
    }

    /**
     * Check whether {@code path} is a SafeTensors file.
     *
     * @param path path to inspect
     * @return true if the file is a valid SafeTensors file
     */
    public static boolean isSafeTensors(Path path) {
        return detect(path).map(f -> f == ModelFormat.SAFETENSORS).orElse(false);
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    private static Optional<ModelFormat> detectByMagic(Path path) {
        try (InputStream in = Files.newInputStream(path)) {
            byte[] header = in.readNBytes(8);
            if (header.length < 4) {
                return Optional.empty();
            }

            // ── GGUF ──────────────────────────────────────────────────────
            int magic = ByteBuffer.wrap(header, 0, 4)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .getInt();
            if (magic == GGUF_MAGIC) {
                return Optional.of(ModelFormat.GGUF);
            }

            // ── SafeTensors ───────────────────────────────────────────────
            if (header.length >= 8) {
                long headerLen = ByteBuffer.wrap(header, 0, 8)
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .getLong();
                if (headerLen >= SAFETENSORS_MIN_HEADER_LEN
                        && headerLen <= SAFETENSORS_MAX_HEADER_LEN) {
                    // Peek at the JSON start to confirm: should be '{'
                    byte[] jsonStart = in.readNBytes(1);
                    if (jsonStart.length == 1 && jsonStart[0] == '{') {
                        return Optional.of(ModelFormat.SAFETENSORS);
                    }
                }
            }

            return Optional.empty();
        } catch (IOException e) {
            // File unreadable — fall through to extension detection
            return Optional.empty();
        }
    }
}
