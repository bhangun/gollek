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
 */
public final class ModelFormatDetector {

    private static final int GGUF_MAGIC = 0x46554747; // 'G','G','U','F'
    private static final long SAFETENSORS_MIN_HEADER_LEN = 8L;
    private static final long SAFETENSORS_MAX_HEADER_LEN = 256L * 1024 * 1024;

    private ModelFormatDetector() {
    }

    public static Optional<ModelFormat> detect(Path path) {
        if (path == null || !Files.exists(path))
            return Optional.empty();

        // 1. If it's a directory, look for marker files
        if (Files.isDirectory(path)) {
            for (ModelFormat format : ModelFormat.values()) {
                for (String marker : format.getMarkerFiles()) {
                    if (Files.exists(path.resolve(marker))) {
                        return Optional.of(format);
                    }
                }
            }
            return Optional.empty();
        }

        // 2. Exact magic detection for regular files
        Optional<ModelFormat> byMagic = detectByMagic(path);
        if (byMagic.isPresent())
            return byMagic;

        // 3. Extension-based fallback
        return detectByExtension(path.getFileName().toString());
    }

    public static Optional<ModelFormat> detectByExtension(String fileName) {
        if (fileName == null || fileName.isBlank())
            return Optional.empty();
        String lower = fileName.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".gguf"))
            return Optional.of(ModelFormat.GGUF);
        if (lower.endsWith(".safetensors") || lower.endsWith(".safetensor"))
            return Optional.of(ModelFormat.SAFETENSORS);
        if (lower.endsWith(".onnx"))
            return Optional.of(ModelFormat.ONNX);
        if (lower.endsWith(".litertlm"))
            return Optional.of(ModelFormat.LITERT);
        return Optional.empty();
    }

    public static boolean isGguf(Path path) {
        return detect(path).map(f -> f == ModelFormat.GGUF).orElse(false);
    }

    public static boolean isSafeTensors(Path path) {
        return detect(path).map(f -> f == ModelFormat.SAFETENSORS).orElse(false);
    }

    /**
     * Detects Stable Diffusion pipeline by checking for UNet + VAE subdirectories.
     * Handles both ONNX variant (vae_decoder/) and safetensors variant (vae/).
     */
    public static boolean isStableDiffusion(Path modelPath) {
        if (modelPath == null) return false;
        try {
            Path dir = Files.isDirectory(modelPath) ? modelPath : modelPath.getParent();
            if (dir == null || !Files.exists(dir)) return false;
            
            boolean exists = Files.exists(dir.resolve("model_index.json"));
            System.err.println("Checking SD at: " + dir + " -> " + exists);
            return exists;
        } catch (Exception e) {
            return false;
        }
    }

    private static Optional<ModelFormat> detectByMagic(Path path) {
        try (InputStream in = Files.newInputStream(path)) {
            byte[] header = in.readNBytes(8);
            if (header.length < 4)
                return Optional.empty();
            int magic = ByteBuffer.wrap(header, 0, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            if (magic == GGUF_MAGIC)
                return Optional.of(ModelFormat.GGUF);
            if (header.length >= 8) {
                long headerLen = ByteBuffer.wrap(header, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
                if (headerLen >= SAFETENSORS_MIN_HEADER_LEN && headerLen <= SAFETENSORS_MAX_HEADER_LEN) {
                    byte[] jsonStart = in.readNBytes(1);
                    if (jsonStart.length == 1 && jsonStart[0] == '{')
                        return Optional.of(ModelFormat.SAFETENSORS);
                }
            }
            return Optional.empty();
        } catch (IOException e) {
            return Optional.empty();
        }
    }
}
