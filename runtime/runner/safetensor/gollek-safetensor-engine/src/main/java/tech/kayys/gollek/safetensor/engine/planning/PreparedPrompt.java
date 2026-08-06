package tech.kayys.gollek.safetensor.engine.planning;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

/**
 * Engine-owned prepared prompt.
 *
 * <p>This makes prompt shaping explicit instead of passing raw strings through the
 * engine. A stable fingerprint gives future backends a place to attach text-prefix
 * cache lookups or shared-prefill reuse policies, similar to the text prefix caching
 * ideas discussed in Barrios (2026), arXiv:2601.19139.
 */
public record PreparedPrompt(
        String formattedPrompt,
        String fingerprint,
        PromptReusePlan reusePlan,
        boolean defaultSystemInjected,
        String modelType) {

    public PreparedPrompt {
        Objects.requireNonNull(formattedPrompt, "formattedPrompt");
        Objects.requireNonNull(fingerprint, "fingerprint");
        Objects.requireNonNull(reusePlan, "reusePlan");
        Objects.requireNonNull(modelType, "modelType");
    }

    public static PreparedPrompt of(String formattedPrompt, boolean defaultSystemInjected, String modelType) {
        String fingerprint = sha256(formattedPrompt);
        return new PreparedPrompt(
                formattedPrompt,
                fingerprint,
                PromptReusePlan.exactPrompt(fingerprint, formattedPrompt.length()),
                defaultSystemInjected,
                modelType == null ? "" : modelType);
    }

    private static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }
}
