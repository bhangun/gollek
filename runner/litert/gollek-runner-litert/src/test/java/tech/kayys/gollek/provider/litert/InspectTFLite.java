package tech.kayys.gollek.provider.litert;
 
 import java.nio.file.Path;
 import java.nio.file.Paths;
 import java.lang.foreign.Arena;
 import java.lang.foreign.MemorySegment;
 import java.nio.channels.FileChannel;
 
 public class InspectTFLite {
     public static void main(String[] args) throws Exception {
         String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-web.task";
         Path p = Paths.get(pathStr);
         if (!java.nio.file.Files.exists(p)) {
             System.out.println("Model file not found: " + pathStr);
             return;
         }
 
         LiteRTContainerParser.ContainerInfo info = LiteRTContainerParser.parse(p);
         System.out.println("---- PARSE INFO ----");
         System.out.println("Format: " + info.format());
         System.out.println("Offset: " + info.tfliteOffset());
         System.out.println("Size: " + info.tfliteSize());
         System.out.println("SubModels: " + info.subModels().size());
         
         System.out.println("---- NATIVE LOAD ----");
         String libraryPath = System.getProperty("LITERT_LIBRARY_PATH");
         if (libraryPath == null) {
             libraryPath = "/Users/bhangun/.gollek/libs/libLiteRt.dylib";
         }
         LiteRTNativeBindings bindings = new LiteRTNativeBindings(Paths.get(libraryPath));
         
         try (Arena arena = Arena.ofConfined()) {
             MemorySegment env = bindings.createEnvironment(arena);
             
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
             
             System.out.println("✓ CompiledModel created!");
             
             int sigs = bindings.getNumModelSignatures(model, arena);
             System.out.println("Signatures: " + sigs);
             for (int i = 0; i < sigs; i++) {
                 MemorySegment sig = bindings.getModelSignature(model, i, arena);
                 System.out.println("  Sig[" + i + "]: " + bindings.getSignatureKey(sig, arena));
                 System.out.println("    Inputs: " + bindings.getNumSignatureInputs(sig, arena));
                 System.out.println("    Outputs: " + bindings.getNumSignatureOutputs(sig, arena));
             }
             
             System.out.println("Subgraphs: " + bindings.getNumModelSubgraphs(model, arena));
 
             bindings.destroyCompiledModel(compiledModel);
             bindings.destroyOptions(opts);
             bindings.destroyModel(model);
             bindings.destroyEnvironment(env);
         }
     }
 }
