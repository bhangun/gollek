package tech.kayys.gollek.sdk.hub;

import java.nio.file.Path;

/**
 * Configuration for model hub operations.
 */
public record HubConfig(
    String revision,
    Path cacheDir,
    String token,
    boolean forceDownload,
    int timeoutSeconds
) {
    public static final HubConfig DEFAULT = new HubConfig(
        "main",
        Path.of(System.getProperty("user.home"), ".gollek", "models"),
        null,
        false,
        300
    );

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String revision = "main";
        private Path cacheDir = Path.of(System.getProperty("user.home"), ".gollek", "models");
        private String token = null;
        private boolean forceDownload = false;
        private int timeoutSeconds = 300;

        public Builder revision(String v) { this.revision = v; return this; }
        public Builder cacheDir(Path v) { this.cacheDir = v; return this; }
        public Builder cacheDir(String v) { this.cacheDir = Path.of(v); return this; }
        public Builder token(String v) { this.token = v; return this; }
        public Builder forceDownload(boolean v) { this.forceDownload = v; return this; }
        public Builder timeoutSeconds(int v) { this.timeoutSeconds = v; return this; }

        public HubConfig build() {
            return new HubConfig(revision, cacheDir, token, forceDownload, timeoutSeconds);
        }
    }
}
