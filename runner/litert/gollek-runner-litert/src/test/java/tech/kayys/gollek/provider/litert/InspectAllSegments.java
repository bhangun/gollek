package tech.kayys.gollek.provider.litert;
 
 import java.nio.file.Path;
 import java.nio.file.Paths;
 import java.util.List;
 import java.lang.foreign.Arena;
 import java.lang.foreign.MemorySegment;
 
 public class InspectAllSegments {
     public static void main(String[] args) throws Exception {
         String pathStr = "/Users/bhangun/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm";
         Path p = Paths.get(pathStr);
         
         System.out.println("---- SCANNING ALL SEGMENTS IN " + pathStr + " ----");
         List<Long> offsets = LiteRTContainerParser.findTfl3SegmentsForInspection(p);
         
         String libraryPath = System.getProperty("LITERT_LIBRARY_PATH");
         if (libraryPath == null) {
             libraryPath = "/Users/bhangun/.gollek/libs/libLiteRt.dylib";
         }
         LiteRTNativeBindings bindings = new LiteRTNativeBindings(Paths.get(libraryPath));
         long fileSize = java.nio.file.Files.size(p);
 
         try (Arena arena = Arena.ofConfined()) {
             MemorySegment env = bindings.createEnvironment(arena);
 
             for (int i = 0; i < offsets.size(); i++) {
                 long off = offsets.get(i);
                 long next = (i + 1 < offsets.size()) ? offsets.get(i + 1) : fileSize;
                 long size = next - off;
                 
                 System.out.println("\nSegment " + i + " at 0x" + Long.toHexString(off) + " (" + size + " bytes):");
                 try {
                     // Note: We'd ideally use createModelFromBuffer for segments, 
                     // but LiteRT 2.0 prefers createModelFromFile.
                     // For inspection of segments within a container, we map the segment memory.
                     java.nio.channels.FileChannel channel = java.nio.channels.FileChannel.open(p);
                     java.nio.MappedByteBuffer mbb = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, off, size);
                     MemorySegment segment = MemorySegment.ofBuffer(mbb);
                     
                     MemorySegment model = bindings.createModelFromBuffer(segment, size, arena);
                     MemorySegment opts = bindings.createOptions(arena);
                     
                     // Use Metal on Mac
                     if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                         int accel = LiteRTNativeBindings.kLiteRtHwAcceleratorCpu | LiteRTNativeBindings.kLiteRtHwAcceleratorGpu;
                         bindings.setOptionsHardwareAccelerators(opts, accel);
                     }
 
                     MemorySegment compiledModel = bindings.createCompiledModel(env, model, opts, arena);
                     System.out.println("  ✓ CompiledModel created!");
                     
                     int sigs = bindings.getNumModelSignatures(model, arena);
                     System.out.println("  Sigs: " + sigs);
                     for (int s = 0; s < sigs; s++) {
                         MemorySegment sig = bindings.getModelSignature(model, s, arena);
                         System.out.println("    Sig[" + s + "]: " + bindings.getSignatureKey(sig, arena));
                     }
                     
                     int subgraphs = bindings.getNumModelSubgraphs(model, arena);
                     System.out.println("  Subgraphs: " + subgraphs);
                     
                     bindings.destroyCompiledModel(compiledModel);
                     bindings.destroyOptions(opts);
                     bindings.destroyModel(model);
                 } catch (Exception e) {
                     System.out.println("  ERROR: " + e.getMessage());
                 }
             }
             bindings.destroyEnvironment(env);
         }
     }
 }
