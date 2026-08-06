package tech.kayys.gollek.sdk.util;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Resolves Gollek runtime home with Gollek-first defaults.
 *
 * <p>Centralizes the public Gollek home contract for the CLI, SDK, runners, and
 * embedding applications. Host products can set {@code gollek.home} or
 * {@code GOLLEK_HOME}; otherwise Gollek uses {@code ~/.gollek}.
 */
public final class GollekHome {
    private static final String GOLLEK_HOME_PROP = "gollek.home";

    private GollekHome() {
    }

    public static Path resolve() {
        String explicit = firstNonBlank(
                System.getProperty(GOLLEK_HOME_PROP),
                System.getenv("GOLLEK_HOME"));
        if (hasText(explicit)) {
            return Paths.get(explicit).toAbsolutePath().normalize();
        }

        String userHome = System.getProperty("user.home");
        return Paths.get(userHome, ".gollek").toAbsolutePath().normalize();
    }

    public static Path path(String... parts) {
        Path root = resolve();
        for (String part : parts) {
            root = root.resolve(part);
        }
        return root;
    }

    public static void applySystemProperties() {
        Path resolved = resolve();
        if (!hasText(System.getProperty(GOLLEK_HOME_PROP))) {
            System.setProperty(GOLLEK_HOME_PROP, resolved.toString());
        }
    }

    private static boolean hasText(String value) {
        return value != null && !value.trim().isEmpty();
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (hasText(value)) {
                return value.trim();
            }
        }
        return null;
    }
}
