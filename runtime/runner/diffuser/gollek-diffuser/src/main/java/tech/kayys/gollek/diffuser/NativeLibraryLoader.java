package tech.kayys.gollek.diffuser;

import java.lang.foreign.*;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Loads the ONNX Runtime shared library via the standard System.loadLibrary
 * path,
 * then validates that a sentinel symbol is resolvable through the FFM Linker.
 */
public final class NativeLibraryLoader {

    private static volatile boolean loaded = false;

    private NativeLibraryLoader() {
    }

    public static synchronized void load() {
        if (loaded)
            return;

        String osName = System.getProperty("os.name").toLowerCase();
        String libName;

        if (osName.contains("win")) {
            libName = "onnxruntime"; // .dll resolved by System
        } else if (osName.contains("mac")) {
            libName = "onnxruntime"; // .dylib
        } else {
            libName = "onnxruntime"; // .so
        }

        // Allow override via system property: -Dsd.ort.lib=/full/path/libonnxruntime.so
        String override = System.getProperty("sd.ort.lib");
        if (override != null) {
            System.load(override);
        } else {
            System.loadLibrary(libName);
        }

        // Verify symbol resolution through FFM Linker
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();
        lookup.find("OrtGetApiBase")
                .orElseThrow(() -> new UnsatisfiedLinkError(
                        "OrtGetApiBase not found – is ONNX Runtime on LD_LIBRARY_PATH?"));

        loaded = true;
        System.out.println("[NativeLibraryLoader] ONNX Runtime loaded successfully.");
    }
}
