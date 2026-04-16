package tech.kayys.gollek.inference.gguf;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Locates, loads, and initialises the llama.cpp native library.
 *
 * <p>Discovery order:
 * <ol>
 *   <li>Explicit file path via config / {@code GOLLEK_LLAMA_LIB_PATH}</li>
 *   <li>Explicit directory via config / {@code GOLLEK_LLAMA_LIB_DIR}</li>
 *   <li>{@link System#loadLibrary} (respects {@code java.library.path})</li>
 *   <li>Common runtime directories ({@code ~/.gollek/source/llama-cpp/lib}, {@code ~/.gollek/libs/llama}, Homebrew, etc.)</li>
 *   <li>Classpath resource extraction to a temp directory</li>
 * </ol>
 *
 * <p>This class is package-private; use {@link LlamaCppBinding#load()} as the entry point.
 */
final class LlamaNativeLoader {

    private static final Logger log = Logger.getLogger(LlamaNativeLoader.class);
    private static final String LIB_BASE_NAME = "llama";

    private LlamaNativeLoader() {}

    /**
     * Loads the native library and returns a {@link SymbolLookup} for it.
     *
     * @param verbose        {@code true} to log detailed load steps
     * @param explicitLibPath optional absolute path to the library file
     * @param explicitLibDir  optional directory containing the library
     * @return a {@link SymbolLookup} backed by the loaded library
     * @throws RuntimeException if the library cannot be found or loaded
     */
    static SymbolLookup load(boolean verbose, Optional<String> explicitLibPath, Optional<String> explicitLibDir) {
        try {
            loadNativeLibrary(explicitLibPath, explicitLibDir, verbose);
            SymbolLookup lookup = SymbolLookup.loaderLookup();
            suppressNativeLogs(lookup);
            if (!verbose) {
                log.info("Loaded llama.cpp native library (quiet mode)");
            }
            return lookup;
        } catch (Throwable e) {
            log.warn("Failed to load llama.cpp library: " + e.getMessage());
            throw new RuntimeException("Failed to load llama.cpp library", e);
        }
    }

    // ── Library discovery ─────────────────────────────────────────────────────

    private static void loadNativeLibrary(Optional<String> explicitLibPath, Optional<String> explicitLibDir,
            boolean verbose) throws Exception {
        String mappedLibName = System.mapLibraryName(LIB_BASE_NAME);
        List<String> attempts = new ArrayList<>();

        Optional<String> configuredLibPath = explicitLibPath.filter(s -> !s.isBlank())
                .or(() -> optionalEnv("GOLLEK_LLAMA_LIB_PATH"));
        Optional<String> configuredLibDir = explicitLibDir.filter(s -> !s.isBlank())
                .or(() -> optionalEnv("GOLLEK_LLAMA_LIB_DIR"));

        // 1) Explicit absolute library file path
        if (configuredLibPath.isPresent()) {
            Path candidate = Path.of(configuredLibPath.get()).toAbsolutePath();
            attempts.add("explicit path: " + candidate);
            if (Files.exists(candidate)) {
                loadWithDependencies(candidate.getParent(), candidate.getFileName().toString(), verbose);
                return;
            }
        }

        // 2) Explicit directory
        if (configuredLibDir.isPresent()) {
            Path dir = Path.of(configuredLibDir.get()).toAbsolutePath();
            Path candidate = dir.resolve(mappedLibName);
            attempts.add("explicit dir: " + candidate);
            if (Files.exists(candidate)) {
                loadWithDependencies(dir, mappedLibName, verbose);
                return;
            }
        }

        // 3) System.loadLibrary (java.library.path / DYLD_LIBRARY_PATH)
        try {
            System.loadLibrary(LIB_BASE_NAME);
            return;
        } catch (Throwable t) {
            attempts.add("System.loadLibrary failed: " + t.getMessage());
        }

        // 4) Common runtime filesystem locations
        for (Path dir : candidateRuntimeDirs()) {
            Path candidate = dir.resolve(mappedLibName);
            attempts.add("runtime dir: " + candidate);
            if (Files.exists(candidate)) {
                loadWithDependencies(dir, mappedLibName, verbose);
                return;
            }
        }

        // 5) Extract from classpath resources
        Path extracted = extractAndLoadFromResources(verbose);
        if (extracted != null) {
            return;
        }

        throw new RuntimeException(
                "Unable to locate llama.cpp native library. Attempts: " + String.join(" | ", attempts));
    }

    // ── Dependency loading ────────────────────────────────────────────────────

    private static void loadWithDependencies(Path libraryDir, String mainLibName, boolean verbose) {
        List<String> errors = new ArrayList<>();
        String shimName = shimLibraryFileName();

        if (isMacOS()) {
            fixMacOSLibraryPaths(libraryDir, verbose);
        }

        // Load ggml base libraries first (strict dependency order)
        String[] baseLibs = {
                "libggml-base.dylib", "libggml-base.0.dylib", "libggml-base.0.9.5.dylib",
                "libggml-cpu.dylib",  "libggml-cpu.0.dylib",  "libggml-cpu.0.9.5.dylib",
                "libggml-blas.dylib", "libggml-blas.0.dylib", "libggml-blas.0.9.5.dylib",
                "libggml-metal.dylib","libggml-metal.0.dylib","libggml-metal.0.9.5.dylib",
                "libggml.dylib",      "libggml.0.dylib",      "libggml.0.9.5.dylib"
        };
        for (String lib : baseLibs) {
            Path p = libraryDir.resolve(lib);
            if (Files.exists(p)) {
                try { System.load(p.toAbsolutePath().toString()); }
                catch (UnsatisfiedLinkError ignored) {}
            }
        }

        // Load remaining dependencies
        for (String dep : dependencyLoadOrder(libraryDir)) {
            if (dep.equals(mainLibName) || dep.startsWith("lib" + LIB_BASE_NAME)
                    || dep.equals(shimName) || dep.startsWith("libggml")) continue;
            Path depPath = libraryDir.resolve(dep);
            if (!Files.exists(depPath)) continue;
            try { System.load(depPath.toAbsolutePath().toString()); }
            catch (UnsatisfiedLinkError e) { errors.add(dep + ": " + e.getMessage()); }
        }

        System.load(libraryDir.resolve(mainLibName).toAbsolutePath().toString());

        Path shimPath = libraryDir.resolve(shimName);
        if (Files.exists(shimPath)) {
            try { System.load(shimPath.toAbsolutePath().toString()); }
            catch (UnsatisfiedLinkError e) { errors.add(shimName + ": " + e.getMessage()); }
        }

        if (!errors.isEmpty()) {
            log.debugf("Some optional dependency loads failed: %s", String.join(" | ", errors));
        }
    }

    private static void fixMacOSLibraryPaths(Path libraryDir, boolean verbose) {
        try (var stream = Files.list(libraryDir)) {
            stream.filter(p -> p.toString().endsWith(".dylib")).forEach(libPath -> {
                try {
                    String libName = libPath.getFileName().toString();
                    ProcessBuilder pb = new ProcessBuilder("install_name_tool",
                            "-change", "@rpath/" + libName, "@loader_path/" + libName,
                            libPath.toAbsolutePath().toString());
                    pb.redirectErrorStream(true);
                    Process proc = pb.start();
                    proc.getInputStream().transferTo(java.io.OutputStream.nullOutputStream());
                    proc.waitFor();
                } catch (Exception e) {
                    if (verbose) log.warnf("Failed to fix library path for %s: %s",
                            libPath.getFileName(), e.getMessage());
                }
            });
        } catch (IOException e) {
            if (verbose) log.warnf("Failed to fix macOS library paths: %s", e.getMessage());
        }
    }

    // ── Resource extraction ───────────────────────────────────────────────────

    private static Path extractAndLoadFromResources(boolean verbose) throws Exception {
        String mainLib = System.mapLibraryName(LIB_BASE_NAME);
        Path tempDir = Files.createTempDirectory("llama-cpp");
        List<Path> extracted = new ArrayList<>();

        for (String fileName : nativeDependencyFileNames()) {
            Path p = extractLibraryFromResources(fileName, tempDir);
            if (p != null) extracted.add(p);
        }

        if (extracted.isEmpty()) return null;

        if (verbose) log.infof("Extracted native libraries to %s", tempDir);
        loadWithDependencies(tempDir, mainLib, verbose);
        return tempDir.resolve(mainLib);
    }

    private static Path extractLibraryFromResources(String fileName, Path targetDir) throws Exception {
        for (String prefix : nativeResourcePrefixes()) {
            try (InputStream is = getResourceAsStream(prefix + fileName)) {
                if (is != null) {
                    Path target = targetDir.resolve(fileName);
                    Files.copy(is, target, StandardCopyOption.REPLACE_EXISTING);
                    if (!isWindows()) target.toFile().setExecutable(true);
                    return target;
                }
            }
        }
        return null;
    }

    private static List<String> nativeResourcePrefixes() {
        boolean hasCuda = System.getenv("CUDA_PATH") != null && !System.getenv("CUDA_PATH").isBlank();
        List<String> prefixes = new ArrayList<>();
        prefixes.add(hasCuda ? "/gollek-gguf/native-libs/cuda/" : "/gollek-gguf/native-libs/cpu/");
        prefixes.add("/gollek-gguf/native-libs/");
        prefixes.add(hasCuda ? "/native-libs/cuda/" : "/native-libs/cpu/");
        prefixes.add("/native-libs/");
        return prefixes;
    }

    // ── Log suppression ───────────────────────────────────────────────────────

    /**
     * Installs a no-op log callback to silence Metal/CUDA pipeline messages.
     * Uses the Gollek shim if available, otherwise falls back to {@code llama_log_set}.
     */
    static void suppressNativeLogs(SymbolLookup lookup) {
        try {
            var shimAddr = lookup.find("gollek_llama_log_disable");
            if (shimAddr.isPresent()) {
                Linker.nativeLinker()
                        .downcallHandle(shimAddr.get(), FunctionDescriptor.ofVoid())
                        .invoke();
                return;
            }
            var logSetAddr = lookup.find("llama_log_set");
            if (logSetAddr.isPresent()) {
                MethodHandle noOp = MethodHandles.lookup().findStatic(
                        LlamaNativeLoader.class, "noOpLogCallback",
                        MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class));
                MemorySegment stub = Linker.nativeLinker().upcallStub(noOp,
                        FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS),
                        Arena.global());
                Linker.nativeLinker()
                        .downcallHandle(logSetAddr.get(),
                                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                        .invoke(stub, MemorySegment.NULL);
            }
        } catch (Throwable ignored) {}
    }

    @SuppressWarnings("unused")
    private static void noOpLogCallback(int level, MemorySegment text, MemorySegment userData) {}

    // ── Utilities ─────────────────────────────────────────────────────────────

    private static List<Path> candidateRuntimeDirs() {
        List<Path> dirs = new ArrayList<>();
        optionalEnv("GOLLEK_LLAMA_LIB_PATH").map(Path::of).map(Path::toAbsolutePath)
                .map(Path::getParent).ifPresent(dirs::add);
        optionalEnv("GOLLEK_LLAMA_LIB_DIR").map(Path::of).map(Path::toAbsolutePath).ifPresent(dirs::add);

        Path sourcePathLlamaLib = Path.of(System.getProperty("user.home"),
                ".gollek", "source", "llama-cpp", "lib").toAbsolutePath();
        if (Files.exists(sourcePathLlamaLib)) dirs.add(sourcePathLlamaLib);

        Path sourcePathBin = Path.of(System.getProperty("user.home"),
                ".gollek", "source", "vendor", "llama.cpp", "build", "bin").toAbsolutePath();
        if (Files.exists(sourcePathBin)) dirs.add(sourcePathBin);

        Path sourcePathRoot = Path.of(System.getProperty("user.home"),
                ".gollek", "source", "vendor", "llama.cpp").toAbsolutePath();
        if (Files.exists(sourcePathRoot)) dirs.add(sourcePathRoot);

        dirs.add(Path.of(System.getProperty("user.home"), ".gollek", "libs", "llama").toAbsolutePath());

        if (isMacOS()) {
            Path brew = Path.of("/opt/homebrew/opt/llama.cpp/lib");
            if (Files.exists(brew)) dirs.add(brew);
            Path intelBrew = Path.of("/usr/local/opt/llama.cpp/lib");
            if (Files.exists(intelBrew)) dirs.add(intelBrew);
        }

        dirs.add(Path.of(System.getProperty("user.home"), ".gollek", "native-libs").toAbsolutePath());
        dirs.add(Path.of(".").toAbsolutePath());
        dirs.add(Path.of("native-libs").toAbsolutePath());
        dirs.add(Path.of("lib").toAbsolutePath());

        ProcessHandle.current().info().command().ifPresent(cmd -> {
            Path parent = Path.of(cmd).toAbsolutePath().getParent();
            if (parent != null) {
                dirs.add(parent);
                dirs.add(parent.resolve("native-libs"));
                dirs.add(parent.resolve("lib"));
            }
        });

        Path cwd = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        for (int i = 0; i < 8 && cwd != null; i++) {
            dirs.add(cwd.resolve("gollek/plugins/runner/gguf/gollek-ext-runner-gguf/src/main/resources/native-libs"));
            dirs.add(cwd.resolve("extension/format/gguf/gollek-ext-runner-gguf/target/llama-cpp/lib"));
            dirs.add(cwd.resolve("inference/format/gguf/source/llama-cpp/llama.cpp/build/bin"));
            cwd = cwd.getParent();
        }
        return dirs;
    }

    private static List<String> dependencyLoadOrder(Path libraryDir) {
        Set<String> names = new LinkedHashSet<>(nativeDependencyFileNames());
        try (var paths = Files.list(libraryDir)) {
            paths.filter(Files::isRegularFile)
                    .map(p -> p.getFileName().toString())
                    .filter(n -> n.endsWith(nativeLibExt()) && (n.startsWith("libggml") || n.startsWith("ggml")))
                    .sorted()
                    .forEach(names::add);
        } catch (IOException ignored) {}
        return new ArrayList<>(names);
    }

    private static List<String> nativeDependencyFileNames() {
        String ext = nativeLibExt();
        List<String> names = new ArrayList<>();
        if (isWindows()) {
            names.addAll(List.of("ggml-base" + ext, "ggml-cpu" + ext, "ggml-blas" + ext,
                    "ggml-cuda" + ext, "ggml-metal" + ext, "ggml" + ext, "llama" + ext,
                    "libggml-base" + ext, "libggml-cpu" + ext, "libggml-blas" + ext,
                    "libggml-cuda" + ext, "libggml-metal" + ext, "libggml" + ext, "libllama" + ext));
        } else {
            names.addAll(List.of(
                    "libggml-base" + ext, "libggml-cpu" + ext, "libggml-blas" + ext,
                    "libggml-cuda" + ext, "libggml-cuda.0" + ext,
                    "libggml-vulkan" + ext, "libggml-vulkan.0" + ext,
                    "libggml-metal" + ext, "libggml-metal.0" + ext,
                    "libggml-opencl" + ext, "libggml-opencl.0" + ext,
                    "libggml" + ext, "libggml.0" + ext,
                    "libggml-base.0" + ext, "libggml-cpu.0" + ext, "libggml-blas.0" + ext,
                    "libllama" + ext, "libllama.0" + ext, shimLibraryFileName()));
        }
        return names;
    }

    private static Optional<String> optionalEnv(String key) {
        String v = System.getenv(key);
        return (v == null || v.isBlank()) ? Optional.empty() : Optional.of(v.trim());
    }

    private static InputStream getResourceAsStream(String path) {
        InputStream is = LlamaNativeLoader.class.getResourceAsStream(path);
        if (is != null) return is;
        String noSlash = path.startsWith("/") ? path.substring(1) : path;
        is = LlamaNativeLoader.class.getClassLoader().getResourceAsStream(noSlash);
        if (is != null) return is;
        ClassLoader tcl = Thread.currentThread().getContextClassLoader();
        return tcl != null ? tcl.getResourceAsStream(noSlash) : null;
    }

    static String shimLibraryFileName() {
        return (isWindows() ? "" : "lib") + "gollek_llama_shim" + nativeLibExt();
    }

    static boolean isMacOS() {
        return System.getProperty("os.name", "").toLowerCase().contains("mac");
    }

    static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase().contains("win");
    }

    static String nativeLibExt() {
        if (isWindows()) return ".dll";
        if (isMacOS())   return ".dylib";
        return ".so";
    }
}
