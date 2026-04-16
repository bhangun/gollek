package tech.kayys.gollek.metal.binding;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import org.jboss.logging.Logger;

/**
 * Utility for finding the libgollek_metal.dylib in standard locations.
 */
public class MetalLibraryDiscovery {
    private static final Logger LOG = Logger.getLogger(MetalLibraryDiscovery.class);
    private static final String LIB_NAME = "libgollek_metal.dylib";

    public static Path findLibrary() {
        // 1. Explicit override
        String override = System.getProperty("gollek.metal.dylib");
        if (override != null) {
            Path p = Path.of(override);
            if (Files.exists(p)) return p;
        }

        // 2. Standard Gollek installation path (~/.gollek/libs)
        String home = System.getProperty("user.home");
        if (home != null) {
            Path p = Path.of(home, ".gollek", "libs", LIB_NAME);
            if (Files.exists(p)) return p;
        }

        // 3. Search in java.library.path
        String libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            for (String dir : libPath.split(File.pathSeparator)) {
                Path p = Path.of(dir, LIB_NAME);
                if (Files.exists(p)) return p;
            }
        }

        // 4. Current directory
        Path p = Path.of(LIB_NAME);
        if (Files.exists(p)) return p;

        return p; 
    }
}
