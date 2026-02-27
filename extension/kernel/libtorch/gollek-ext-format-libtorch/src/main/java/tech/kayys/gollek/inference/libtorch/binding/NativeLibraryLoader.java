package tech.kayys.gollek.inference.libtorch.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Platform-aware native LibTorch library loader.
 * <p>
 * Search order:
 * <ol>
 * <li>Explicit config path ({@code libtorch.provider.native.library-path})</li>
 * <li>Environment variable {@code GOLEK_LIBTORCH_LIB_PATH} (via config) or
 * {@code LIBTORCH_PATH}</li>
 * <li>System library path ({@code java.library.path})</li>
 * <li>Common platform-specific install locations, including
 * {@code ~/.gollek/source/vendor/libtorch}</li>
 * </ol>
 * <p>
 * Thread-safe: uses double-checked locking for one-time initialization.
 */
public final class NativeLibraryLoader {

    private static final Logger log = Logger.getLogger(NativeLibraryLoader.class);

    private static volatile SymbolLookup cachedLookup;
    private static volatile boolean loaded = false;
    private static volatile Throwable loadFailure;

    private NativeLibraryLoader() {
    }

    /**
     * Load the LibTorch native libraries and return a SymbolLookup.
     *
     * @param configuredPath optional explicit path to the libtorch shared library
     *                       directory
     * @return SymbolLookup for resolving native symbols
     * @throws UnsatisfiedLinkError if libraries cannot be found
     */
    public static SymbolLookup load(Optional<String> configuredPath) {
        if (loaded) {
            if (loadFailure != null) {
                throw new RuntimeException("LibTorch load previously failed: " + loadFailure.getMessage(), loadFailure);
            }
            return cachedLookup;
        }

        synchronized (NativeLibraryLoader.class) {
            if (loaded) {
                if (loadFailure != null) {
                    throw new RuntimeException(
                            "LibTorch load previously failed: " + loadFailure.getMessage(),
                            loadFailure);
                }
                return cachedLookup;
            }

            try {
                cachedLookup = doLoad(configuredPath);
                loaded = true;
                log.debug("LibTorch native libraries loaded successfully");
                return cachedLookup;
            } catch (Throwable t) {
                loadFailure = t;
                loaded = true;
                log.errorf(t, "Failed to load LibTorch native libraries");
                throw new RuntimeException("Failed to load LibTorch: " + t.getMessage(), t);
            }
        }
    }

    /**
     * Check if native libraries were loaded successfully.
     */
    public static boolean isLoaded() {
        return loaded && loadFailure == null;
    }

    /**
     * Get the load failure if any.
     */
    public static Optional<Throwable> getLoadFailure() {
        return Optional.ofNullable(loadFailure);
    }

    private static SymbolLookup doLoad(Optional<String> configuredPath) {
        // 1. Try configured path
        if (configuredPath.isPresent()) {
            Path libDir = Path.of(configuredPath.get());
            if (Files.isDirectory(libDir)) {
                log.debugf("Loading LibTorch from configured path: %s", libDir);
                return loadFromDirectory(libDir);
            }
            log.warnf("Configured LibTorch path does not exist: %s", libDir);
        }

        // 2. Try LIBTORCH_PATH environment variable
        String envPath = System.getenv("LIBTORCH_PATH");
        if (envPath != null && !envPath.isBlank()) {
            Path libDir = Path.of(envPath);
            if (Files.isDirectory(libDir)) {
                log.debugf("Loading LibTorch from LIBTORCH_PATH: %s", libDir);
                return loadFromDirectory(libDir);
            }
            log.warnf("LIBTORCH_PATH does not exist: %s", libDir);
        }

        // 3. Try system library path via System.loadLibrary
        try {
            log.debug("Attempting to load LibTorch from system library path");
            System.loadLibrary("torch");
            System.loadLibrary("torch_cpu");
            System.loadLibrary("c10");
            return SymbolLookup.loaderLookup();
        } catch (UnsatisfiedLinkError e) {
            log.debugf("System library path load failed: %s", e.getMessage());
        }

        // 4. Try common platform locations
        Throwable lastCandidateFailure = null;
        List<Path> candidatesWithLibraries = new ArrayList<>();
        for (String candidate : getPlatformCandidates()) {
            if (candidate == null || candidate.isBlank()) {
                continue;
            }
            Path candidatePath = Path.of(candidate).toAbsolutePath();
            if (Files.isDirectory(candidatePath)) {
                if (hasAnyLibrary(candidatePath)) {
                    candidatesWithLibraries.add(candidatePath);
                }
            }
        }

        for (Path candidatePath : candidatesWithLibraries) {
            try {
                cachedLookup = loadFromDirectory(candidatePath);
                return cachedLookup;
            } catch (Throwable t) {
                lastCandidateFailure = t;
            }
        }

        if (lastCandidateFailure != null) {
            throw new RuntimeException(
                    "LibTorch candidate directories were found, but loading failed. Last error: "
                            + lastCandidateFailure.getMessage(),
                    lastCandidateFailure);
        }

        if (!candidatesWithLibraries.isEmpty()) {
            throw new UnsatisfiedLinkError(
                    "LibTorch candidate directories were found but no loadable symbols were resolved");
        }

        throw new UnsatisfiedLinkError(
                "LibTorch native libraries not found. Set GOLEK_LIBTORCH_LIB_PATH or LIBTORCH_PATH "
                        + "or configure libtorch.provider.native.library-path. "
                        + "Source vendor default: ~/.gollek/source/vendor/libtorch");
    }

    private static SymbolLookup loadFromDirectory(Path libDir) {
        maybeClearMacQuarantine(libDir);
        Path libPath = resolveLibraryFile(libDir);
        if (libPath == null) {
            throw new UnsatisfiedLinkError("No LibTorch shared library found in: " + libDir);
        }

        // Load dependencies first
        preloadDependencies(libDir);

        // process
        log.debugf("Loading main LibTorch library: %s", libPath);
        try {
            System.load(libPath.toAbsolutePath().toString());
        } catch (UnsatisfiedLinkError e) {
            log.errorf("Failed to load main library %s: %s", libPath, e.getMessage());
            throw e;
        }

        // Return a lookup that can see all loaded libraries
        return SymbolLookup.loaderLookup();
    }

    private static boolean hasAnyLibrary(Path libDir) {
        return resolveLibraryFile(libDir) != null;
    }

    private static void maybeClearMacQuarantine(Path libDir) {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (!(os.contains("mac") || os.contains("darwin"))) {
            return;
        }

        Path xattr = Path.of("/usr/bin/xattr");
        if (!Files.isExecutable(xattr)) {
            return;
        }

        try {
            Process process = new ProcessBuilder(
                    xattr.toString(), "-dr", "com.apple.quarantine", libDir.toAbsolutePath().toString())
                    .redirectErrorStream(true)
                    .start();
            int exit = process.waitFor();
            if (exit == 0) {
                log.debugf("Cleared macOS quarantine flags for LibTorch path: %s", libDir);
            } else {
                log.debugf("xattr quarantine cleanup exited with code %d for %s", exit, libDir);
            }
        } catch (Exception e) {
            log.debugf("Unable to clear macOS quarantine for %s: %s", libDir, e.getMessage());
        }
    }

    private static void preloadDependencies(Path libDir) {
        String os = System.getProperty("os.name", "").toLowerCase();
        String[] deps;

        if (os.contains("mac") || os.contains("darwin")) {
            deps = new String[] {
                    "libomp.dylib",
                    "libtorch_global_deps.dylib",
                    "libc10.dylib",
                    "libtorch_cpu.dylib"
            };
        } else if (os.contains("win")) {
            deps = new String[] { "c10.dll", "torch_cpu.dll" };
        } else {
            deps = new String[] { "libtorch_global_deps.so", "libc10.so", "libtorch_cpu.so" };
        }

        for (String dep : deps) {
            Path depPath = libDir.resolve(dep);
            if (Files.isRegularFile(depPath)) {
                try {
                    System.load(depPath.toAbsolutePath().toString());
                } catch (Throwable t) {
                    log.debugf("Failed to preload dependency %s (might be normal if already loaded): %s", depPath,
                            t.getMessage());
                }
            }
        }
    }

    private static Path resolveLibraryFile(Path libDir) {
        String os = System.getProperty("os.name", "").toLowerCase();

        String[] libNames;
        if (os.contains("mac") || os.contains("darwin")) {
            libNames = new String[] {
                    "libtorch_wrapper.dylib", "libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib",
                    "lib/libtorch.dylib"
            };
        } else if (os.contains("win")) {
            libNames = new String[] {
                    "torch_wrapper.dll", "torch.dll", "torch_cpu.dll", "c10.dll",
                    "lib/torch.dll"
            };
        } else {
            // Linux and others
            libNames = new String[] {
                    "libtorch_wrapper.so", "libtorch.so", "libtorch_cpu.so", "libc10.so",
                    "lib/libtorch.so"
            };
        }

        for (String name : libNames) {
            Path candidate = libDir.resolve(name);
            if (Files.isRegularFile(candidate)) {
                log.debugf("Found library candidate: %s", candidate);
                return candidate;
            } else {
                log.tracef("Library candidate not found: %s", candidate);
            }
        }
        return null;
    }

    private static String[] getPlatformCandidates() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String vendorBase = resolveSourceVendorPath();
        String userHome = System.getProperty("user.home");
        String userVendorBase = Path.of(userHome, ".gollek", "source", "vendor", "libtorch").toString();
        String envSource = System.getenv("GOLEK_LIBTORCH_SOURCE_DIR");

        if (os.contains("mac") || os.contains("darwin")) {
            return new String[] {
                    // Project-local build path for development.
                    Path.of(System.getProperty("user.dir", "."), "extension", "kernel", "libtorch", "build", "lib").toString(),
                    Path.of(System.getProperty("user.dir", "."), "inference-gollek", "extension", "kernel", "libtorch", "build", "lib").toString(),
                    Path.of(System.getProperty("user.dir", "."), "extension", "kernel", "libtorch", "src", "main", "resources",
                            "native", "Darwin", "arm64").toString(),
                    Path.of(System.getProperty("user.dir", "."), "inference-gollek", "extension", "kernel", "libtorch", "src",
                            "main", "resources", "native", "Darwin", "arm64").toString(),
                    envSource != null && !envSource.isBlank() ? Path.of(envSource).resolve("lib").toString() : "",
                    envSource != null && !envSource.isBlank() ? envSource : "",
                    userVendorBase + "/lib",
                    vendorBase + "/libtorch-macos/lib",
                    vendorBase + "/libtorch-macos",
                    "/usr/local/lib",
                    "/opt/homebrew/lib",
                    "/opt/libtorch/lib",
                    userHome + "/libtorch/lib"
            };
        } else if (os.contains("win")) {
            return new String[] {
                    envSource != null && !envSource.isBlank() ? Path.of(envSource).resolve("lib").toString() : "",
                    envSource != null && !envSource.isBlank() ? envSource : "",
                    userVendorBase + "/lib",
                    vendorBase + "/libtorch-windows/lib",
                    "C:\\libtorch\\lib",
                    System.getenv("LOCALAPPDATA") + "\\libtorch\\lib"
            };
        } else {
            return new String[] {
                    envSource != null && !envSource.isBlank() ? Path.of(envSource).resolve("lib").toString() : "",
                    envSource != null && !envSource.isBlank() ? envSource : "",
                    userVendorBase + "/lib",
                    vendorBase + "/libtorch-linux/lib",
                    "/usr/local/lib",
                    "/usr/lib",
                    "/opt/libtorch/lib",
                    userHome + "/libtorch/lib"
            };
        }
    }

    private static String resolveSourceVendorPath() {
        String configured = System.getenv("GOLEK_LIBTORCH_SOURCE_DIR");
        if (configured != null && !configured.isBlank()) {
            return configured.trim();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "source", "vendor", "libtorch").toString();
    }
}
