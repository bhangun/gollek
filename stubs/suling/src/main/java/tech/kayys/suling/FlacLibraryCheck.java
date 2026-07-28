package tech.kayys.suling;

/**
 * Fallback probe used when the native Suling module is not checked out.
 */
public final class FlacLibraryCheck {
    private FlacLibraryCheck() {
    }

    public static boolean isAvailable() {
        return false;
    }

    public static String getVersion() {
        return "unavailable";
    }

    public static String getDiagnostics() {
        return "Suling fallback stub";
    }
}
