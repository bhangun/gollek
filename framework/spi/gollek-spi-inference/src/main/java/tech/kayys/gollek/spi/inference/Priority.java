package tech.kayys.gollek.spi.inference;

import java.util.Locale;

/**
 * Standard priority levels for inference requests.
 * Lower levels are processed first (ascending level order).
 */
public enum Priority {
    CRITICAL(0),
    HIGH(1),
    NORMAL(2),
    LOW(3);

    private final int level;

    Priority(int level) {
        this.level = level;
    }

    public int level() {
        return level;
    }

    /**
     * Map string representation to Priority.
     * Case-insensitive, defaults to NORMAL if null or unknown.
     */
    public static Priority fromString(String value) {
        if (value == null) {
            return NORMAL;
        }
        try {
            return Priority.valueOf(value.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            return NORMAL;
        }
    }
}
