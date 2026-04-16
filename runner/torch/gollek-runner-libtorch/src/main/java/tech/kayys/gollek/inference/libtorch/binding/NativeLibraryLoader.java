package tech.kayys.gollek.inference.libtorch.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.StandardCopyOption;
import java.util.zip.CRC32;
import java.util.zip.CheckedInputStream;

/**
 * Platform-aware native LibTorch library loader.
 * <p>
 * Search order:
 * <ol>
 * <li>Explicit config path ({@code libtorch.provider.native.library-path})</li>
 * <li>Environment variable {@code GOLLEK_LIBTORCH_LIB_PATH} (via config) or
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

    private static final String GOLLEK_LIBS_DIR = ".gollek/libs/libtorch";
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
        // First, Find and Pre-load LibTorch dependencies (c10, torch_cpu)
        // These are HEAVY and must come from the system/environment, not the JAR.
        Path libTorchHome = findLibTorchHome(configuredPath);
        if (libTorchHome != null) {
            log.infof("Found LibTorch installation at: %s", libTorchHome);
            preloadDependencies(libTorchHome);
        }

        // 1. Try explicit bundled resource extraction (Highest Priority for stability)
        try {
            Path extractedPath = extractFromResources();
            if (extractedPath != null) {
                if (libTorchHome != null) {
                    fixMacNativeDependencies(extractedPath, libTorchHome);
                }
                log.infof("Successfully extracted and verified bundled library: %s", extractedPath);
                return loadFile(extractedPath);
            }
        } catch (Exception e) {
            log.errorf("Failed to extract bundled library: %s", e.getMessage());
            // Fall through to other methods
        }

        // 2. Try configured path
        if (configuredPath.isPresent()) {
            Path libDir = Path.of(configuredPath.get());
            if (Files.isDirectory(libDir)) {
                log.debugf("Loading LibTorch from configured path: %s", libDir);
                return loadFromDirectory(libDir);
            }
            log.infof("Configured LibTorch path does not exist: %s", libDir);
        }

        // 3. Try LIBTORCH_PATH environment variable
        String envPath = System.getenv("LIBTORCH_PATH");
        if (envPath != null && !envPath.isBlank()) {
            Path libDir = Path.of(envPath);
            if (Files.isDirectory(libDir)) {
                log.debugf("Loading LibTorch from LIBTORCH_PATH: %s", libDir);
                return loadFromDirectory(libDir);
            }
            log.infof("LIBTORCH_PATH does not exist: %s", libDir);
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

        // Try to load torch_wrapper from the found LibTorch home if we didn't use the
        // bundle
        if (libTorchHome != null) {
            try {
                Path libPath = resolveLibraryFile(libTorchHome);
                if (libPath != null && libPath.getFileName().toString().contains("torch_wrapper")) {
                    return loadFile(libPath);
                }
            } catch (Throwable t) {
                lastCandidateFailure = t;
            }
        }

        // 5. Try system library path via System.loadLibrary (Last Resort)
        try {
            log.debug("Attempting to load LibTorch from system library path as last resort");
            System.loadLibrary("torch_wrapper"); // Try our wrapper first
            return SymbolLookup.loaderLookup();
        } catch (UnsatisfiedLinkError e) {
            // ignore
        }

        if (lastCandidateFailure != null) {
            throw new RuntimeException(
                    "LibTorch candidate directories were found, but loading failed. Last error: "
                            + lastCandidateFailure.getMessage(),
                    lastCandidateFailure);
        }

        throw new UnsatisfiedLinkError(
                "LibTorch native libraries not found. Set GOLLEK_LIBTORCH_LIB_PATH or LIBTORCH_PATH "
                        + "or configure libtorch.provider.native.library-path. "
                        + "Source vendor default: ~/.gollek/source/vendor/libtorch");
    }

    private static Path extractFromResources() throws IOException {
        String libName = System.mapLibraryName("torch_wrapper");
        if (libName.contains("liblib")) {
            libName = libName.replace("liblib", "lib");
        }

        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.equals("x86_64"))
            arch = "amd64";
        if (arch.equals("aarch64"))
            arch = "arm64";

        String platformDir;
        if (os.contains("mac") || os.contains("darwin"))
            platformDir = "Darwin";
        else if (os.contains("win"))
            platformDir = "Windows";
        else
            platformDir = "Linux";

        String resourcePath = String.format("/native/%s/%s/%s", platformDir, arch, libName);
        log.infof("Attempting to load bundled resource from: %s (Class Location: %s)", resourcePath,
                NativeLibraryLoader.class.getProtectionDomain().getCodeSource().getLocation());

        InputStream is = NativeLibraryLoader.class.getResourceAsStream(resourcePath);

        if (is == null) {
            log.infof("Resource not found at %s, trying without arch prefix...", resourcePath);
            resourcePath = String.format("/native/%s/%s", platformDir, libName);
            is = NativeLibraryLoader.class.getResourceAsStream(resourcePath);
        }

        if (is == null) {
            log.warnf("No bundled resource found in JAR at expected paths.");
            return null;
        }
        is.close(); // Just checking for existence here

        Path targetDir = Path.of(System.getProperty("user.home")).resolve(GOLLEK_LIBS_DIR);
        Files.createDirectories(targetDir);
        Path targetFile = targetDir.resolve(libName);

        // Calculate checksum of bundled resource
        long bundledChecksum = calculateResourceChecksum(resourcePath);

        if (Files.exists(targetFile)) {
            long existingChecksum = calculateFileChecksum(targetFile);
            if (bundledChecksum == existingChecksum) {
                log.debugf("Extracted library matches bundled version (checksum: %X)", bundledChecksum);
                return targetFile;
            }
            log.infof("Checksum mismatch for %s. Re-extracting (Bundled: %X, Existing: %X)", libName, bundledChecksum,
                    existingChecksum);
        } else {
            log.infof("Extracting bundled library: %s (Checksum: %X)", libName, bundledChecksum);
        }

        // Re-open stream for extraction
        try (InputStream input = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            Files.copy(input, targetFile, StandardCopyOption.REPLACE_EXISTING);
        }

        maybeClearMacQuarantine(targetFile);
        return targetFile;
    }

    private static long calculateResourceChecksum(String resourcePath) throws IOException {
        try (InputStream is = NativeLibraryLoader.class.getResourceAsStream(resourcePath);
                CheckedInputStream cis = new CheckedInputStream(is, new CRC32())) {
            byte[] buffer = new byte[8192];
            while (cis.read(buffer) != -1)
                ;
            return cis.getChecksum().getValue();
        }
    }

    private static long calculateFileChecksum(Path file) throws IOException {
        try (InputStream is = Files.newInputStream(file);
                CheckedInputStream cis = new CheckedInputStream(is, new CRC32())) {
            byte[] buffer = new byte[8192];
            while (cis.read(buffer) != -1)
                ;
            return cis.getChecksum().getValue();
        }
    }

    private static void fixMacNativeDependencies(Path wrapperPath, Path libTorchHome) {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (!(os.contains("mac") || os.contains("darwin"))) {
            return;
        }

        String[] deps = { "libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib" };
        for (String dep : deps) {
            Path depPath = libTorchHome.resolve(dep);
            if (Files.exists(depPath)) {
                try {
                    log.debugf("Fixing macOS dependency: %s -> %s", dep, depPath);
                    Process process = new ProcessBuilder(
                            "install_name_tool",
                            "-change",
                            "@loader_path/" + dep,
                            depPath.toAbsolutePath().toString(),
                            wrapperPath.toAbsolutePath().toString())
                            .redirectErrorStream(true)
                            .start();
                    int exit = process.waitFor();
                    if (exit != 0) {
                        log.warnf("install_name_tool failed with code %d for %s", exit, dep);
                    }
                } catch (Exception e) {
                    log.debugf("Failed to run install_name_tool for %s: %s", dep, e.getMessage());
                }
            }
        }
    }

    private static Path findLibTorchHome(Optional<String> configuredPath) {
        // 1. Check configured path
        if (configuredPath.isPresent()) {
            Path path = Path.of(configuredPath.get());
            if (Files.isDirectory(path) && hasCoreLibraries(path))
                return path;
        }

        // 2. Check LIBTORCH_PATH
        String envPath = System.getenv("LIBTORCH_PATH");
        if (envPath != null && !envPath.isBlank()) {
            Path path = Path.of(envPath);
            if (Files.isDirectory(path) && hasCoreLibraries(path))
                return path;
        }

        // 3. Check common candidates
        for (String candidate : getPlatformCandidates()) {
            if (candidate == null || candidate.isBlank())
                continue;
            Path path = Path.of(candidate);
            if (Files.isDirectory(path) && hasCoreLibraries(path))
                return path;
        }

        return null;
    }

    private static boolean hasCoreLibraries(Path libDir) {
        String os = System.getProperty("os.name", "").toLowerCase();
        String c10 = (os.contains("win")) ? "c10.dll"
                : (os.contains("mac") || os.contains("darwin")) ? "libc10.dylib" : "libc10.so";
        return Files.exists(libDir.resolve(c10));
    }

    private static SymbolLookup loadFile(Path libPath) {
        log.debugf("Loading LibTorch from file: %s", libPath);
        try {
            System.load(libPath.toAbsolutePath().toString());
            return SymbolLookup.loaderLookup();
        } catch (UnsatisfiedLinkError e) {
            log.errorf("Failed to load library %s: %s", libPath, e.getMessage());
            throw e;
        }
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
            deps = new String[] { "c10.dll", "libtorch_cpu.dll" };
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
                    "torch_wrapper.dylib", "libtorch_wrapper.dylib", "libtorch.dylib", "libtorch_cpu.dylib",
                    "libc10.dylib",
                    "lib/libtorch.dylib"
            };
        } else if (os.contains("win")) {
            libNames = new String[] {
                    "torch_wrapper.dll", "libtorch_wrapper.dll", "libtorch.dll", "libtorch_cpu.dll", "c10.dll",
                    "lib/libtorch.dll"
            };
        } else {
            // Linux and others
            libNames = new String[] {
                    "torch_wrapper.so", "libtorch_wrapper.so", "libtorch.so", "libtorch_cpu.so", "libc10.so",
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

        // Standard Gollek native library location
        String standardLibPath = Path.of(userHome, ".gollek", "libs", "libtorch").toString();

        // Legacy source vendor path
        String userVendorBase = Path.of(userHome, ".gollek", "source", "vendor", "libtorch").toString();
        String envSource = System.getenv("GOLLEK_LIBTORCH_SOURCE_DIR");

        if (os.contains("mac") || os.contains("darwin")) {
            return new String[] {
                    // Standard Gollek location (highest priority after explicit config)
                    standardLibPath,

                    // Project-local build path for development.
                    Path.of(System.getProperty("user.dir", "."), "gollek", "plugins", "runner", "torch", "build", "lib")
                            .toString(),
                    Path.of(System.getProperty("user.dir", "."), "gollek", "plugins", "runner", "torch", "src", "main",
                            "resources",
                            "native", "Darwin", "arm64").toString(),
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
                    standardLibPath,
                    envSource != null && !envSource.isBlank() ? Path.of(envSource).resolve("lib").toString() : "",
                    envSource != null && !envSource.isBlank() ? envSource : "",
                    userVendorBase + "/lib",
                    vendorBase + "/libtorch-windows/lib",
                    "C:\\libtorch\\lib",
                    System.getenv("LOCALAPPDATA") + "\\libtorch\\lib"
            };
        } else {
            return new String[] {
                    standardLibPath,
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
        String configured = System.getenv("GOLLEK_LIBTORCH_SOURCE_DIR");
        if (configured != null && !configured.isBlank()) {
            return configured.trim();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "source", "vendor", "libtorch").toString();
    }
}
