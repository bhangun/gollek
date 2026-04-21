package tech.kayys.gollek.cli.util;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.DecimalFormat;

public class CLIUtils {

    /**
     * Format byte size to human readable string (KB, MB, GB, TB).
     */
    public static String formatSize(long bytes) {
        if (bytes <= 0) return "0 B";
        final String[] units = new String[]{"B", "KB", "MB", "GB", "TB"};
        int digitGroups = (int) (Math.log10(bytes) / Math.log10(1024));
        return new DecimalFormat("#,##0.#").format(bytes / Math.pow(1024, digitGroups)) + " " + units[digitGroups];
    }

    /**
     * Generate a short 6-character hex ID from a string.
     */
    public static String generateShortId(String input) {
        if (input == null || input.isBlank()) return "unknown";
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(input.getBytes());
            StringBuilder hexString = new StringBuilder();
            for (int i = 0; i < 3; i++) { // Extract first 3 bytes (6 chars)
                String hex = Integer.toHexString(0xff & hash[i]);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            // Fallback: simple checksum
            return String.format("%06x", input.hashCode() & 0xFFFFFF);
        }
    }
}
