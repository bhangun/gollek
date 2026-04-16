package tech.kayys.gollek.provider.litert;
 
 import java.nio.file.Path;
 import java.nio.file.Paths;
 import java.lang.foreign.Arena;
 import java.lang.foreign.MemorySegment;
 
 public class DeepInspectTask {
     public static void main(String[] args) throws Exception {
         String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-web.task";
         Path p = Paths.get(pathStr);
         String libraryPath = System.getProperty("LITERT_LIBRARY_PATH");
         if (libraryPath == null) {
             libraryPath = "/Users/bhangun/.gollek/libs/libLiteRt.dylib";
         }
         
         LiteRTNativeBindings bindings = new LiteRTNativeBindings(Paths.get(libraryPath));
         try (Arena arena = Arena.ofConfined()) {
             System.out.println("Deep Inspecting: " + pathStr);
             
             // 1. Create Environment
             MemorySegment env = bindings.createEnvironment(arena);
             
             // 2. Create Model
             MemorySegment model = bindings.createModelFromFile(pathStr, arena);
             
             // 3. Create Options
             MemorySegment opts = bindings.createOptions(arena);
             
             // LiteRT 2.0 hardware selection
             if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                 int accel = LiteRTNativeBindings.kLiteRtHwAcceleratorCpu | LiteRTNativeBindings.kLiteRtHwAcceleratorGpu;
                 bindings.setOptionsHardwareAccelerators(opts, accel);
                 System.out.println("Accelerators set: CPU + GPU (Metal)");
             }
 
             // 4. Create Compiled Model
             MemorySegment compiledModel = bindings.createCompiledModel(env, model, opts, arena);
             System.out.println("✓ CompiledModel Created!");
             
             int numSigs = bindings.getNumModelSignatures(model, arena);
             System.out.println("Signatures: " + numSigs);
             for (int i = 0; i < numSigs; i++) {
                 MemorySegment sig = bindings.getModelSignature(model, i, arena);
                 String key = bindings.getSignatureKey(sig, arena);
                 int numInputs = bindings.getNumSignatureInputs(sig, arena);
                 int numOutputs = bindings.getNumSignatureOutputs(sig, arena);
                 System.out.println("  Sig[" + i + "]: '" + key + "' (" + numInputs + " inputs, " + numOutputs + " outputs)");
                 
                 for (int j = 0; j < numInputs; j++) {
                     System.out.println("    Input[" + j + "]: " + bindings.getSignatureInputName(sig, j, arena));
                 }
                 for (int j = 0; j < numOutputs; j++) {
                     System.out.println("    Output[" + j + "]: " + bindings.getSignatureOutputName(sig, j, arena));
                 }
             }
 
             int subgraphs = bindings.getNumModelSubgraphs(model, arena);
             System.out.println("Subgraphs: " + subgraphs);
             
             // 5. Cleanup
             bindings.destroyCompiledModel(compiledModel);
             bindings.destroyOptions(opts);
             bindings.destroyModel(model);
             bindings.destroyEnvironment(env);
         }
     }
 }
