package tech.kayys.gollek.provider.litert;
 
 import org.junit.jupiter.api.Test;
 
 import java.io.InputStream;
 import java.net.URL;
 import java.nio.file.Files;
 import java.nio.file.Path;
 import java.nio.file.StandardCopyOption;
 
 import java.lang.foreign.Arena;
 import java.lang.foreign.MemorySegment;
 import java.lang.foreign.ValueLayout;
 
 import static org.junit.jupiter.api.Assertions.assertNotNull;
 import static org.junit.jupiter.api.Assertions.assertTrue;
 import static org.junit.jupiter.api.Assumptions.assumeTrue;
 
 /**
  * Integration tests for LiteRT 2.0 Native Bindings using the CompiledModel pipeline.
  */
 class LiteRTNativeBindingsIntegrationTest {
 
     @Test
     void loadsRealModelAndRunsInference() throws Exception {
         String libraryPath = resolveLibraryPath();
         assumeTrue(libraryPath != null && !libraryPath.isBlank(),
                 "Set LITERT_LIBRARY_PATH or litert.library-path to run this test");
         Path libPath = Path.of(libraryPath);
         assumeTrue(Files.exists(libPath) && Files.isRegularFile(libPath),
                 "LiteRT native library missing: " + libraryPath);
 
         Path modelPath = resolveModelPath();
         assumeTrue(modelPath != null && Files.exists(modelPath) && Files.size(modelPath) > 0,
                 "Set GOLLEK_TFLITE_MODEL_PATH or GOLLEK_TFLITE_MODEL_URL to a valid .litertlm model");
 
         try (Arena arena = Arena.ofConfined()) {
             LiteRTNativeBindings bindings = new LiteRTNativeBindings(libPath);
 
             // 1. Create Environment
             MemorySegment env = bindings.createEnvironment(arena);
             assertNotNull(env);
             assertTrue(env.address() != 0);
 
             // 2. Create Model
             MemorySegment model = bindings.createModelFromFile(modelPath.toString(), arena);
             assertNotNull(model);
 
             // 3. Create Options
             MemorySegment options = bindings.createOptions(arena);
             assertNotNull(options);
 
             // 4. Create Compiled Model
             MemorySegment compiledModel = null;
             try {
                 compiledModel = bindings.createCompiledModel(env, model, options, arena);
                 assertNotNull(compiledModel);
                 assertTrue(compiledModel.address() != 0);
 
                 // 5. Basic Introspection
                 int numSigs = bindings.getNumModelSignatures(model, arena);
                 assertTrue(numSigs >= 0);
 
                 // 6. Test I/O flow (simplified)
                 // Getting buffer requirements and creating tensor buffers
                 // This validates the FFM bindings for the new 2.0 API
                 MemorySegment inputReq = bindings.getCompiledModelInputBufferRequirements(compiledModel, 0, 0, arena);
                 MemorySegment outputReq = bindings.getCompiledModelOutputBufferRequirements(compiledModel, 0, 0, arena);
                 
                 assertNotNull(inputReq);
                 assertNotNull(outputReq);
 
                 MemorySegment inputBuf = bindings.createManagedTensorBufferFromRequirements(env, MemorySegment.NULL, inputReq, arena);
                 MemorySegment outputBuf = bindings.createManagedTensorBufferFromRequirements(env, MemorySegment.NULL, outputReq, arena);
                 
                 assertNotNull(inputBuf);
                 assertNotNull(outputBuf);
                 
                 // Cleanup managed buffers
                 bindings.destroyTensorBuffer(inputBuf);
                 bindings.destroyTensorBuffer(outputBuf);
 
             } finally {
                 if (compiledModel != null && compiledModel.address() != 0) {
                     bindings.destroyCompiledModel(compiledModel);
                 }
                 bindings.destroyOptions(options);
                 bindings.destroyModel(model);
                 bindings.destroyEnvironment(env);
             }
         }
     }
 
     private static Path resolveModelPath() throws Exception {
         String modelPath = envOrProp("GOLLEK_TFLITE_MODEL_PATH", "gollek.litertmodel.path");
         if (modelPath != null && !modelPath.isBlank()) {
             Path path = Path.of(modelPath);
             if (Files.exists(path)) {
                 return path;
             }
         }
 
         String modelUrl = envOrProp("GOLLEK_TFLITE_MODEL_URL", "gollek.litertmodel.url");
         if (modelUrl == null || modelUrl.isBlank()) {
             return null;
         }
 
         Path cacheDir = resolveCacheDir();
         Files.createDirectories(cacheDir);
         String fileName = modelUrl.substring(modelUrl.lastIndexOf('/') + 1);
         if (fileName.isBlank()) {
             fileName = "model.litertlm";
         }
         Path target = cacheDir.resolve(fileName);
         if (Files.exists(target) && Files.size(target) > 0) {
             return target;
         }
 
         boolean required = isRequired();
         try (InputStream in = new URL(modelUrl).openStream()) {
             Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
         } catch (Exception e) {
             if (!required) {
                 assumeTrue(false, "Failed to download LiteRT model: " + e.getClass().getSimpleName());
             }
             throw e;
         }
         return target;
     }
 
     private static Path resolveCacheDir() {
         String configured = envOrProp("GOLLEK_TFLITE_CACHE_DIR", "gollek.litertcache-dir");
         if (configured != null && !configured.isBlank()) {
             return Path.of(configured);
         }
         return Path.of(System.getProperty("user.home"), ".gollek", "cache", "litert");
     }
 
     private static String resolveLibraryPath() {
         String configured = envOrProp("LITERT_LIBRARY_PATH", "litert.library-path");
         if (configured != null && !configured.isBlank()) {
             return configured;
         }
         String os = System.getProperty("os.name", "").toLowerCase();
         // LiteRT 2.0 uses libLiteRt (new name) or libtensorflowlite_c (legacy)
         String[] libNames = os.contains("linux") ? new String[]{"libLiteRt.so", "libtensorflowlite_c.so"}
                 : os.contains("mac") ? new String[]{"libLiteRt.dylib", "libtensorflowlite_c.dylib"}
                         : os.contains("win") ? new String[]{"LiteRt.dll", "tensorflowlite_c.dll"}
                                 : new String[0];
 
         for (String libName : libNames) {
             Path fallback = Path.of(System.getProperty("user.home"), ".gollek", "libs", libName);
             if (Files.exists(fallback)) {
                 return fallback.toString();
             }
         }
         return null;
     }
 
     private static boolean isRequired() {
         return Boolean.parseBoolean(envOrProp("GOLLEK_TFLITE_REQUIRED", "gollek.litertrequired", "false"));
     }
 
     private static String envOrProp(String envKey, String propKey) {
         return envOrProp(envKey, propKey, null);
     }
 
     private static String envOrProp(String envKey, String propKey, String fallback) {
         String env = System.getenv(envKey);
         if (env != null && !env.isBlank()) {
             return env.trim();
         }
         String prop = System.getProperty(propKey);
         if (prop != null && !prop.isBlank()) {
             return prop.trim();
         }
         return fallback;
     }
 }
