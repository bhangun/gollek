package tech.kayys.gollek.spi.model;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Shared model utilities.
 */
public final class ModelUtils {

    private ModelUtils() {}

    /**
     * Generate a stable 6-character short ID from a full model ID.
     */
    public static String generateShortId(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return "unknown";
        }
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(modelId.getBytes());
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 3; i++) {
                sb.append(String.format("%02x", hash[i]));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            // Fallback to simple hashCode prefix if SHA-256 is missing
            return String.format("%06x", modelId.hashCode() & 0xFFFFFF);
        }
    }
}
