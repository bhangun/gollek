package tech.kayys.gollek.provider.litert;
 
 import java.nio.file.Path;
 import java.nio.file.Paths;
 import java.lang.foreign.Arena;
 import java.lang.foreign.MemorySegment;
 
 public class InspectTFLite3 {
     public static void main(String[] args) throws Exception {
         String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-web.task";
         Path p = Paths.get(pathStr);
         if (!java.nio.file.Files.exists(p)) {
             System.out.println("Model file not found: " + pathStr);
             return;
         }
 
         String libraryPath = System.getProperty("LITERT_LIBRARY_PATH");
         if (libraryPath == null) {
             libraryPath = "/Users/bhangun/.gollek/libs/libLiteRt.dylib";
         }
         LiteRTNativeBindings bindings = new LiteRTNativeBindings(Paths.get(libraryPath));
         
         try (Arena arena = Arena.ofConfined()) {
             MemorySegment env = bindings.createEnvironment(arena);
             MemorySegment model = bindings.createModelFromFile(p.toString(), arena);
             MemorySegment opts = bindings.createOptions(arena);
             MemorySegment compiledModel = bindings.createCompiledModel(env, model, opts, arena);
             
             System.out.println("✓ CompiledModel created successfully!");
             System.out.println("Fully Accelerated: " + bindings.isFullyAccelerated(compiledModel, arena));
 
             bindings.destroyCompiledModel(compiledModel);
             bindings.destroyOptions(opts);
             bindings.destroyModel(model);
             bindings.destroyEnvironment(env);
         }
     }
 }
