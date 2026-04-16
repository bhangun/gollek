package tech.kayys.gollek.provider.litert;
 
 import java.nio.file.Path;
 import java.nio.file.Paths;
 import java.lang.foreign.Arena;
 import java.lang.foreign.MemorySegment;
 import java.nio.channels.FileChannel;
 
 public class InspectFullModel {
     public static void main(String[] args) throws Exception {
         String[] models = {
             "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-web.task",
             "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm"
         };
         
         String libraryPath = System.getProperty("LITERT_LIBRARY_PATH");
         if (libraryPath == null) {
             libraryPath = "/Users/bhangun/.gollek/libs/libLiteRt.dylib";
         }
         LiteRTNativeBindings bindings = new LiteRTNativeBindings(Paths.get(libraryPath));
         
         try (Arena arena = Arena.ofConfined()) {
             MemorySegment env = bindings.createEnvironment(arena);
 
             for (String pathStr : models) {
                 System.out.println("\n========================================");
                 System.out.println("INSPECTING: " + pathStr);
                 try {
                     Path p = Paths.get(pathStr);
                     if (!java.nio.file.Files.exists(p)) {
                         System.out.println("File not found: " + pathStr);
                         continue;
                     }
                     
                     LiteRTContainerParser.ContainerInfo info = LiteRTContainerParser.parse(p);
                     System.out.println("Format: " + info.format() + ", Offset: " + info.tfliteOffset() + ", Size: " + info.tfliteSize());
 
                     MemorySegment model;
                     if (info.tfliteOffset() > 0) {
                         FileChannel channel = FileChannel.open(p);
                         java.nio.MappedByteBuffer mbb = channel.map(FileChannel.MapMode.READ_ONLY, info.tfliteOffset(), info.tfliteSize());
                         MemorySegment segment = MemorySegment.ofBuffer(mbb);
                         model = bindings.createModelFromBuffer(segment, info.tfliteSize(), arena);
                     } else {
                         model = bindings.createModelFromFile(p.toString(), arena);
                     }
                         
                     MemorySegment opts = bindings.createOptions(arena);
                     MemorySegment compiledModel = bindings.createCompiledModel(env, model, opts, arena);
                     
                     System.out.println("✓ CompiledModel created successfully!");
                     
                     int sigCount = bindings.getNumModelSignatures(model, arena);
                     System.out.println("Signatures: " + sigCount);
                     for (int i = 0; i < sigCount; i++) {
                         MemorySegment sig = bindings.getModelSignature(model, i, arena);
                         String key = bindings.getSignatureKey(sig, arena);
                         System.out.println("  Sig[" + i + "]: " + key);
                         
                         int inputs = bindings.getNumSignatureInputs(sig, arena);
                         int outputs = bindings.getNumSignatureOutputs(sig, arena);
                         System.out.println("    Inputs: " + inputs + ", Outputs: " + outputs);
                         for (int j = 0; j < inputs; j++) {
                             System.out.println("      In[" + j + "]: " + bindings.getSignatureInputName(sig, j, arena));
                         }
                     }
                     
                     bindings.destroyCompiledModel(compiledModel);
                     bindings.destroyOptions(opts);
                     bindings.destroyModel(model);
                 } catch (Exception e) {
                     System.out.println("ERROR inspecting " + pathStr + ": " + e.getMessage());
                 }
             }
             bindings.destroyEnvironment(env);
         }
     }
 }
