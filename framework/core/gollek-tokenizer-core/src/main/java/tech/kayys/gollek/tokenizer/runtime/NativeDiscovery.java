package tech.kayys.gollek.tokenizer.runtime;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

/**
 * Utility to discover native libraries used by the tokenizer (e.g., spm_bridge).
 */
public class NativeDiscovery {

    /**
     * Finds the spm_bridge native library.
     * 
     * <p>Search order:
     * <ol>
     *   <li>{@code GOLLEK_NATIVE_LIB} environment variable</li>
     *   <li>{@code gollek.native.lib} system property</li>
     *   <li>{@code ~/.gollek/native/} directory</li>
     *   <li>Current working directory</li>
     * </ol>
     */
    public static Optional<Path> findSpmBridge() {
        String libName = System.getProperty("os.name").toLowerCase().contains("win") 
            ? "spm_bridge.dll" 
            : (System.getProperty("os.name").toLowerCase().contains("mac") ? "libspm_bridge.dylib" : "libspm_bridge.so");

        // 1. Env var
        String env = System.getenv("GOLLEK_NATIVE_LIB");
        if (env != null) {
            Path p = Paths.get(env).resolve(libName);
            if (Files.exists(p)) return Optional.of(p);
        }

        // 2. System property
        String prop = System.getProperty("gollek.native.lib");
        if (prop != null) {
            Path p = Paths.get(prop).resolve(libName);
            if (Files.exists(p)) return Optional.of(p);
        }

        // 3. ~/.gollek/native/
        Path home = Paths.get(System.getProperty("user.home")).resolve(".gollek").resolve("native").resolve(libName);
        if (Files.exists(home)) return Optional.of(home);

        // 4. Current dir
        Path local = Paths.get(libName);
        if (Files.exists(local)) return Optional.of(local);

        return Optional.empty();
    }
}
