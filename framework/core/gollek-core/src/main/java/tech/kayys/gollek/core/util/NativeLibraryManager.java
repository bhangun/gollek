package tech.kayys.gollek.core.util;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Centralized native library management for Gollek inference engines.
 */
public class NativeLibraryManager {

    private static final Logger log = Logger.getLogger(NativeLibraryManager.class.getName());
    private static final String DEFAULT_BASE_DIR = Path.of(System.getProperty("user.home"), ".gollek", "libs")
            .toString();

    private final String libraryName;
    private final String libraryBaseName;
    private final Path baseDir;
    private Path installedLibraryPath;

    public NativeLibraryManager(String libraryName) {
        this.libraryName = libraryName;
        this.libraryBaseName = normalizeLibraryName(libraryName);
        this.baseDir = resolveBaseDirectory();
    }

    public static Path getBaseDirectory() {
        return Path.of(resolveBaseDirectoryFromEnv());
    }

    private static String resolveBaseDirectoryFromEnv() {
        String gollekNativeDir = System.getenv("GOLLEK_NATIVE_LIB_DIR");
        if (gollekNativeDir != null && !gollekNativeDir.isBlank()) {
            log.fine("Using GOLLEK_NATIVE_LIB_DIR: " + gollekNativeDir);
            return gollekNativeDir.trim();
        }
        log.fine("Using default native lib directory: " + DEFAULT_BASE_DIR);
        return DEFAULT_BASE_DIR;
    }

    private Path resolveBaseDirectory() {
        String gollekNativeDir = System.getenv("GOLLEK_NATIVE_LIB_DIR");
        if (gollekNativeDir != null && !gollekNativeDir.isBlank()) {
            return Path.of(gollekNativeDir.trim());
        }
        return Path.of(DEFAULT_BASE_DIR);
    }

    public Path getLibraryDirectory() {
        Path libDir = baseDir.resolve(libraryName);
        if (!Files.exists(libDir)) {
            try {
                Files.createDirectories(libDir);
                log.fine("Created native library directory: " + libDir);
            } catch (IOException e) {
                log.log(Level.SEVERE, "Failed to create native library directory " + libDir, e);
            }
        }
        return libDir;
    }

    public Path getLibraryFilePath() {
        String libFileName = getLibraryFileName();
        return getLibraryDirectory().resolve(libFileName);
    }

    public String getLibraryFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin")) {
            return "lib" + libraryBaseName + ".dylib";
        } else if (os.contains("win")) {
            return libraryBaseName + ".dll";
        } else {
            return "lib" + libraryBaseName + ".so";
        }
    }

    public static String getNativeExtension() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin")) {
            return ".dylib";
        } else if (os.contains("win")) {
            return ".dll";
        } else {
            return ".so";
        }
    }

    public Path copyToStandardLocation(Path sourcePath) throws IOException {
        if (!Files.exists(sourcePath)) {
            throw new IOException("Source library does not exist: " + sourcePath);
        }

        Path destDir = getLibraryDirectory();
        Path destPath = destDir.resolve(sourcePath.getFileName());

        Files.createDirectories(destDir);
        Files.copy(sourcePath, destPath, StandardCopyOption.REPLACE_EXISTING);

        if (!isWindows()) {
            destPath.toFile().setExecutable(true);
        }

        log.info("Copied native library " + sourcePath.getFileName() + " to standard location: " + destPath);
        this.installedLibraryPath = destPath;
        return destPath;
    }

    public List<Path> copyLibrariesToStandardLocation(Path sourceDir, List<String> libraryNames) throws IOException {
        List<Path> copiedPaths = new ArrayList<>();
        Path destDir = getLibraryDirectory();
        Files.createDirectories(destDir);

        for (String libName : libraryNames) {
            String fileName = normalizeLibraryName(libName);
            String fullFileName = "lib" + fileName + getNativeExtension();
            Path sourcePath = sourceDir.resolve(fullFileName);

            if (Files.exists(sourcePath)) {
                Path destPath = destDir.resolve(fullFileName);
                Files.copy(sourcePath, destPath, StandardCopyOption.REPLACE_EXISTING);

                if (!isWindows()) {
                    destPath.toFile().setExecutable(true);
                }

                copiedPaths.add(destPath);
                log.fine("Copied " + fullFileName + " to " + destPath);
            } else {
                log.fine("Library not found: " + sourcePath);
            }
        }

        if (!copiedPaths.isEmpty()) {
            log.info("Copied " + copiedPaths.size() + " native libraries to " + destDir);
        }

        return copiedPaths;
    }

    public boolean loadLibrary() {
        Path libPath = getLibraryFilePath();
        if (!Files.exists(libPath)) {
            log.warning("Native library not found at " + libPath);
            return false;
        }

        try {
            System.load(libPath.toAbsolutePath().toString());
            log.info("Loaded native library: " + libPath);
            return true;
        } catch (UnsatisfiedLinkError e) {
            log.log(Level.SEVERE, "Failed to load native library " + libPath, e);
            return false;
        }
    }

    public static boolean loadLibrary(Path libPath) {
        if (!Files.exists(libPath)) {
            log.warning("Native library not found at " + libPath);
            return false;
        }

        try {
            System.load(libPath.toAbsolutePath().toString());
            log.info("Loaded native library: " + libPath);
            return true;
        } catch (UnsatisfiedLinkError e) {
            log.log(Level.SEVERE, "Failed to load native library " + libPath, e);
            return false;
        }
    }

    public boolean libraryExists() {
        return Files.exists(getLibraryFilePath());
    }

    public Optional<Path> getInstalledLibraryPath() {
        return Optional.ofNullable(installedLibraryPath);
    }

    public static void clearMacQuarantine(Path libDir) {
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
                log.fine("Cleared macOS quarantine flags for: " + libDir);
            } else {
                log.fine("xattr quarantine cleanup exited with code " + exit + " for " + libDir);
            }
        } catch (Exception e) {
            log.log(Level.FINE, "Unable to clear macOS quarantine for " + libDir, e);
        }
    }

    private static String normalizeLibraryName(String name) {
        if (name.startsWith("lib")) {
            name = name.substring(3);
        }
        if (name.endsWith(".dylib") || name.endsWith(".so") || name.endsWith(".dll")) {
            name = name.substring(0, name.lastIndexOf('.'));
        }
        return name;
    }

    private static boolean isWindows() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("win");
    }
}
