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
 * <li>Explicit config path ({@code libtorch.provider.native-lib.library-path})</li>
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

    private static final String GOLLEK_LIBS_SUBDIR = "libs/libtorch";
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
                ensurePlatformCompatibility(libDir, "Configured LibTorch path");
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
                ensurePlatformCompatibility(libDir, "LIBTORCH_PATH");
                log.debugf("Loading LibTorch from LIBTORCH_PATH: %s", libDir);
                return loadFromDirectory(libDir);
            }
            log.infof("LIBTORCH_PATH does not exist: %s", libDir);
        }

        // 4. Try common platform locations
        Throwable lastCandidateFailure = null;
        List<Path> candidatesWithLibraries = new ArrayList<>();
        List<String> wrongPlatformCandidates = new ArrayList<>();
        for (String candidate : getPlatformCandidates()) {
            if (candidate == null || candidate.isBlank()) {
                continue;
            }
            Path candidatePath = Path.of(candidate).toAbsolutePath();
            if (Files.isDirectory(candidatePath)) {
                String mismatch = detectPlatformMismatch(candidatePath);
                if (mismatch != null) {
                    wrongPlatformCandidates.add(candidatePath + " (" + mismatch + ")");
                    continue;
                }
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

        for (Path candidatePath : candidatesWithLibraries) {
            try {
                log.debugf("Attempting to load LibTorch from platform candidate: %s", candidatePath);
                return loadFromDirectory(candidatePath);
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

        if (!wrongPlatformCandidates.isEmpty()) {
            throw new UnsatisfiedLinkError(
                    "Found LibTorch directories, but they target a different platform: "
                            + String.join("; ", wrongPlatformCandidates));
        }

        throw new UnsatisfiedLinkError(
                "LibTorch native libraries not found. Set GOLLEK_LIBTORCH_LIB_PATH or LIBTORCH_PATH "
                        + "or configure libtorch.provider.native-lib.library-path. "
                        + "Source vendor default: " + resolveSourceVendorPath());
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

        Path targetDir = resolveGollekHome().resolve(GOLLEK_LIBS_SUBDIR);
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
            Path depPath = resolveDependencyFile(libTorchHome, dep);
            if (depPath != null && Files.exists(depPath)) {
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
            if (Files.isDirectory(path)) {
                ensurePlatformCompatibility(path, "Configured LibTorch path");
            }
            if (Files.isDirectory(path) && hasCoreLibraries(path))
                return path;
        }

        // 2. Check LIBTORCH_PATH
        String envPath = System.getenv("LIBTORCH_PATH");
        if (envPath != null && !envPath.isBlank()) {
            Path path = Path.of(envPath);
            if (Files.isDirectory(path)) {
                ensurePlatformCompatibility(path, "LIBTORCH_PATH");
            }
            if (Files.isDirectory(path) && hasCoreLibraries(path))
                return path;
        }

        // 3. Check common candidates
        for (String candidate : getPlatformCandidates()) {
            if (candidate == null || candidate.isBlank())
                continue;
            Path path = Path.of(candidate);
            if (Files.isDirectory(path) && detectPlatformMismatch(path) != null) {
                continue;
            }
            if (Files.isDirectory(path) && hasCoreLibraries(path))
                return path;
        }

        return null;
    }

    private static boolean hasCoreLibraries(Path libDir) {
        return resolveDependencyFile(libDir, currentPlatform().c10Library()) != null;
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
        ensurePlatformCompatibility(libDir, "LibTorch directory");
        Path libPath = resolveLibraryFile(libDir);
        if (libPath == null) {
            throw new UnsatisfiedLinkError("No LibTorch shared library found in: " + libDir);
        }

        // Load dependencies first
        preloadDependencies(libPath.getParent());

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
        String[] deps = currentPlatform().preloadDependencies();

        for (String dep : deps) {
            Path depPath = resolveDependencyFile(libDir, dep);
            if (depPath != null && Files.isRegularFile(depPath)) {
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
        return resolveFirstExisting(libDir, currentPlatform().libraryCandidates());
    }

    private static String[] getPlatformCandidates() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String userHome = System.getProperty("user.home");
        String vendorBase = resolveSourceVendorPath();
        String defaultVendorBase = Path.of(userHome, ".gollek", "source", "vendor", "libtorch").toString();
        Path gollekHome = resolveGollekHome();

        // Standard Gollek native library location
        String standardLibPath = gollekHome.resolve(GOLLEK_LIBS_SUBDIR).toString();
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
                    vendorBase + "/lib",
                    vendorBase + "/libtorch-macos/lib",
                    vendorBase + "/libtorch-macos",
                    defaultVendorBase + "/lib",
                    defaultVendorBase + "/libtorch-macos/lib",
                    defaultVendorBase + "/libtorch-macos",
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
                    vendorBase + "/lib",
                    defaultVendorBase + "/lib",
                    vendorBase + "/libtorch-windows/lib",
                    "C:\\libtorch\\lib",
                    System.getenv("LOCALAPPDATA") + "\\libtorch\\lib"
            };
        } else {
            return new String[] {
                    standardLibPath,
                    envSource != null && !envSource.isBlank() ? Path.of(envSource).resolve("lib").toString() : "",
                    envSource != null && !envSource.isBlank() ? envSource : "",
                    vendorBase + "/lib",
                    defaultVendorBase + "/lib",
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
        return resolveGollekHome().resolve("source").resolve("vendor").resolve("libtorch").toString();
    }

    private static Path resolveGollekHome() {
        String configuredProperty = System.getProperty("gollek.home");
        if (configuredProperty != null && !configuredProperty.isBlank()) {
            return Path.of(configuredProperty.trim()).toAbsolutePath();
        }

        String configuredEnv = System.getenv("GOLLEK_HOME");
        if (configuredEnv != null && !configuredEnv.isBlank()) {
            return Path.of(configuredEnv.trim()).toAbsolutePath();
        }

        return Path.of(System.getProperty("user.home")).resolve(".gollek");
    }

    private static void ensurePlatformCompatibility(Path libDir, String sourceLabel) {
        String mismatch = detectPlatformMismatch(libDir);
        if (mismatch != null) {
            throw new UnsatisfiedLinkError(sourceLabel + " is not compatible with " + currentPlatform().displayName()
                    + ": " + mismatch);
        }
    }

    private static String detectPlatformMismatch(Path libDir) {
        List<String> sharedLibraries = listSharedLibraries(libDir);
        if (sharedLibraries.isEmpty()) {
            return null;
        }

        String expectedExtension = currentPlatform().sharedLibraryExtension();
        boolean hasCompatibleLibrary = sharedLibraries.stream().anyMatch(name -> name.endsWith(expectedExtension));
        if (hasCompatibleLibrary) {
            return null;
        }

        return "found " + String.join(", ", sharedLibraries) + " but expected " + expectedExtension
                + " libraries. " + currentPlatform().recommendedArchiveHint();
    }

    private static List<String> listSharedLibraries(Path libDir) {
        List<String> result = new ArrayList<>();
        collectSharedLibraries(libDir, "", result);
        collectSharedLibraries(libDir.resolve("lib"), "lib/", result);
        return result;
    }

    private static void collectSharedLibraries(Path directory, String prefix, List<String> result) {
        if (!Files.isDirectory(directory)) {
            return;
        }

        try (var stream = Files.list(directory)) {
            stream.filter(Files::isRegularFile)
                    .map(path -> path.getFileName().toString())
                    .filter(NativeLibraryLoader::isSharedLibraryName)
                    .map(name -> prefix + name)
                    .sorted()
                    .forEach(result::add);
        } catch (IOException e) {
            log.debugf("Unable to inspect LibTorch directory %s: %s", directory, e.getMessage());
        }
    }

    private static boolean isSharedLibraryName(String fileName) {
        return fileName.endsWith(".dylib") || fileName.endsWith(".so") || fileName.endsWith(".dll");
    }

    private static Path resolveDependencyFile(Path libDir, String fileName) {
        return resolveFirstExisting(libDir, new String[] { fileName });
    }

    private static Path resolveFirstExisting(Path libDir, String[] fileNames) {
        for (String fileName : fileNames) {
            Path direct = libDir.resolve(fileName);
            if (Files.isRegularFile(direct)) {
                log.debugf("Found library candidate: %s", direct);
                return direct;
            }

            Path nested = libDir.resolve("lib").resolve(fileName);
            if (Files.isRegularFile(nested)) {
                log.debugf("Found library candidate: %s", nested);
                return nested;
            }

            log.tracef("Library candidate not found: %s or %s", direct, nested);
        }
        return null;
    }

    private static PlatformSpec currentPlatform() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        boolean arm64 = arch.contains("aarch64") || arch.contains("arm64");

        if (os.contains("mac") || os.contains("darwin")) {
            return new PlatformSpec(
                    "macOS" + (arm64 ? " Apple Silicon" : ""),
                    ".dylib",
                    "libc10.dylib",
                    new String[] {
                            "libomp.dylib",
                            "libtorch_global_deps.dylib",
                            "libc10.dylib",
                            "libtorch_cpu.dylib"
                    },
                    new String[] {
                            "torch_wrapper.dylib",
                            "libtorch_wrapper.dylib",
                            "libtorch.dylib",
                            "libtorch_cpu.dylib",
                            "libc10.dylib"
                    },
                    arm64
                            ? "Download a macOS arm64 archive such as https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip"
                            : "Download a macOS LibTorch archive instead of a Linux .so bundle");
        }

        if (os.contains("win")) {
            return new PlatformSpec(
                    "Windows",
                    ".dll",
                    "c10.dll",
                    new String[] { "c10.dll", "libtorch_cpu.dll" },
                    new String[] {
                            "torch_wrapper.dll",
                            "libtorch_wrapper.dll",
                            "libtorch.dll",
                            "libtorch_cpu.dll",
                            "c10.dll"
                    },
                    "Download a Windows LibTorch archive instead of Linux/macOS binaries");
        }

        return new PlatformSpec(
                "Linux",
                ".so",
                "libc10.so",
                new String[] { "libtorch_global_deps.so", "libc10.so", "libtorch_cpu.so" },
                new String[] {
                        "torch_wrapper.so",
                        "libtorch_wrapper.so",
                        "libtorch.so",
                        "libtorch_cpu.so",
                        "libc10.so"
                },
                "Download a Linux LibTorch archive instead of macOS/Windows binaries");
    }

    private record PlatformSpec(
            String displayName,
            String sharedLibraryExtension,
            String c10Library,
            String[] preloadDependencies,
            String[] libraryCandidates,
            String recommendedArchiveHint) {
    }
}
